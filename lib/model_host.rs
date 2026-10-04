//! OpenAI-compatible HTTP transport for a single native model.
//!
//! A dedicated worker owns the model and interleaves requests between decode
//! steps. This provides continuous scheduling, not a GPU/vLLM execution backend.
use crate::native::{
    FinishReason, GenerateRequest, GenerateResponse, NativeBackend, NativeEngine, NativeError,
    SamplingParams,
};
use crate::native_prompt::ConstraintSpec;
use axum::{
    extract::{rejection::JsonRejection, DefaultBodyLimit, Request, State},
    http::{header, StatusCode},
    middleware::{self, Next},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    convert::Infallible,
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;

type SseItem = Result<Event, Infallible>;

#[derive(Clone, Copy)]
pub enum ChatTemplate {
    ChatMl,
    Llama3,
}

#[derive(Clone)]
pub struct HostConfig {
    pub model_name: String,
    pub batch_size: usize,
    /// Total admitted plus waiting requests, not just the decode batch.
    pub max_pending_requests: usize,
    pub max_input_tokens: usize,
    pub max_output_tokens: usize,
    pub max_context_tokens: Option<usize>,
    pub request_timeout: Duration,
    pub api_key: Option<String>,
    /// Explicit template selection avoids silently mismatching a checkpoint.
    pub chat_template: Option<ChatTemplate>,
}
impl Default for HostConfig {
    fn default() -> Self {
        Self {
            model_name: "lighter".into(),
            batch_size: 4,
            max_pending_requests: 64,
            max_input_tokens: 4096,
            max_output_tokens: 512,
            max_context_tokens: None,
            request_timeout: Duration::from_secs(300),
            api_key: None,
            chat_template: None,
        }
    }
}
impl HostConfig {
    pub fn validate(&self) -> Result<(), NativeError> {
        if self.model_name.trim().is_empty()
            || self.batch_size == 0
            || self.max_pending_requests == 0
            || self.max_input_tokens == 0
            || self.max_output_tokens == 0
            || self.request_timeout.is_zero()
            || Instant::now().checked_add(self.request_timeout).is_none()
            || self.max_context_tokens == Some(0)
            || self.api_key.as_ref().is_some_and(|key| key.is_empty())
        {
            return Err(NativeError(
                "model name, limits, timeout and any configured API key must be nonempty/positive"
                    .into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct ApiError {
    status: StatusCode,
    message: String,
}
impl ApiError {
    fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }
    fn bad(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, message)
    }
    fn value(&self) -> Value {
        json!({"error":{"message":self.message,"type":if self.status.is_server_error() {"server_error"} else {"invalid_request_error"},"param":null,"code":self.status.as_u16()}})
    }
}
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(self.value())).into_response()
    }
}

#[derive(Default)]
struct Metrics {
    ready: AtomicBool,
    inflight: AtomicUsize,
    active: AtomicUsize,
    queued: AtomicUsize,
    completed: AtomicU64,
    failed: AtomicU64,
}
impl Metrics {
    fn release(&self) {
        let mut current = self.inflight.load(Ordering::Acquire);
        while current != 0 {
            match self.inflight.compare_exchange_weak(
                current,
                current - 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return,
                Err(actual) => current = actual,
            }
        }
    }
    fn reserve(&self, limit: usize) -> bool {
        let mut current = self.inflight.load(Ordering::Acquire);
        loop {
            if current >= limit {
                return false;
            }
            match self.inflight.compare_exchange_weak(
                current,
                current + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return true,
                Err(actual) => current = actual,
            }
        }
    }
}
struct WorkerReady(Arc<Metrics>);
impl Drop for WorkerReady {
    fn drop(&mut self) {
        self.0.ready.store(false, Ordering::Release);
        self.0.active.store(0, Ordering::Relaxed);
        self.0.queued.store(0, Ordering::Relaxed);
        self.0.inflight.store(0, Ordering::Relaxed);
    }
}

#[derive(Clone)]
struct HostState {
    config: Arc<HostConfig>,
    commands: mpsc::Sender<Command>,
    metrics: Arc<Metrics>,
    ids: Arc<AtomicU64>,
}

/// Build a router with one long-lived backend and a dedicated decode worker.
/// Dropping the router and all in-flight requests closes the worker's channel.
pub fn router<B: NativeBackend + Send + 'static>(
    backend: B,
    eos: Vec<u32>,
    config: HostConfig,
) -> Result<Router, NativeError> {
    config.validate()?;
    let engine = NativeEngine::new(backend, config.batch_size, eos)?;
    let (commands, receiver) = mpsc::channel(config.max_pending_requests);
    let metrics = Arc::new(Metrics::default());
    metrics.ready.store(true, Ordering::Release);
    let state = HostState {
        config: Arc::new(config),
        commands,
        metrics,
        ids: Arc::new(AtomicU64::new(0)),
    };
    let worker_state = state.clone();
    // Do not retain a sender on the worker itself: otherwise shutdown never occurs.
    let worker_config = worker_state.config.clone();
    let worker_metrics = worker_state.metrics.clone();
    drop(worker_state);
    std::thread::Builder::new()
        .name("lighter-decode".into())
        .spawn(move || worker(engine, receiver, worker_config, worker_metrics))
        .map_err(|error| NativeError(format!("cannot start model worker: {error}")))?;
    let api = Router::new()
        .route("/v1/models", get(models))
        .route("/v1/completions", post(completions))
        .route("/v1/graph/completions", post(graph_completions))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/metrics", get(metrics_endpoint))
        .route_layer(middleware::from_fn_with_state(state.clone(), authenticate));
    Ok(Router::new()
        .merge(api)
        .route("/health", get(health))
        .fallback(|| async { ApiError::new(StatusCode::NOT_FOUND, "unknown endpoint") })
        .layer(DefaultBodyLimit::max(1024 * 1024))
        .with_state(state))
}

/// Serve until Ctrl-C, allowing axum to drain existing connections.
pub async fn serve<B: NativeBackend + Send + 'static>(
    listener: tokio::net::TcpListener,
    backend: B,
    eos: Vec<u32>,
    config: HostConfig,
) -> std::io::Result<()> {
    let app = router(backend, eos, config).map_err(std::io::Error::other)?;
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
}

async fn authenticate(State(state): State<HostState>, request: Request, next: Next) -> Response {
    if let Some(key) = &state.config.api_key {
        let supplied = request
            .headers()
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "));
        if !supplied.is_some_and(|supplied| same_key(supplied.as_bytes(), key.as_bytes())) {
            return ApiError::new(
                StatusCode::UNAUTHORIZED,
                "invalid or missing bearer API key",
            )
            .into_response();
        }
    }
    next.run(request).await
}
fn same_key(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter()
        .zip(b)
        .fold(0u8, |difference, (a, b)| difference | (a ^ b))
        == 0
}
async fn health(State(state): State<HostState>) -> Response {
    if state.metrics.ready.load(Ordering::Acquire) {
        Json(json!({"status":"ok","model":state.config.model_name})).into_response()
    } else {
        ApiError::new(StatusCode::SERVICE_UNAVAILABLE, "model worker unavailable").into_response()
    }
}
async fn models(State(state): State<HostState>) -> Json<Value> {
    Json(
        json!({"object":"list","data":[{"id":state.config.model_name,"object":"model","created":0,"owned_by":"lighter"}]}),
    )
}
async fn metrics_endpoint(State(state): State<HostState>) -> String {
    let m = &state.metrics;
    format!("lighter_ready {}\nlighter_requests_inflight {}\nlighter_requests_active {}\nlighter_requests_queued {}\nlighter_requests_completed_total {}\nlighter_requests_failed_total {}\n",
        usize::from(m.ready.load(Ordering::Acquire)), m.inflight.load(Ordering::Relaxed), m.active.load(Ordering::Relaxed),
        m.queued.load(Ordering::Relaxed), m.completed.load(Ordering::Relaxed), m.failed.load(Ordering::Relaxed))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Message {
    role: String,
    content: String,
}
#[derive(Deserialize)]
#[serde(untagged)]
enum Stop {
    One(String),
    Many(Vec<String>),
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StreamOptions {
    #[serde(default)]
    include_usage: bool,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ResponseFormat {
    #[serde(rename = "type")]
    kind: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ApiRequest {
    model: String,
    prompt: Option<String>,
    messages: Option<Vec<Message>>,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    min_p: Option<f32>,
    seed: Option<u64>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    repetition_penalty: Option<f32>,
    stop: Option<Stop>,
    n: Option<usize>,
    #[serde(default)]
    stream: bool,
    stream_options: Option<StreamOptions>,
    response_format: Option<ResponseFormat>,
}
async fn completions(
    State(state): State<HostState>,
    body: Result<Json<ApiRequest>, JsonRejection>,
) -> Response {
    generate(state, body, false).await
}
async fn chat_completions(
    State(state): State<HostState>,
    body: Result<Json<ApiRequest>, JsonRejection>,
) -> Response {
    generate(state, body, true).await
}

/// Graph requests use the same admission, sampling, streaming and cancellation machinery.
async fn graph_completions(
    State(state): State<HostState>,
    body: Result<Json<serde_json::Value>, JsonRejection>,
) -> Response {
    let value = match body {
        Ok(Json(value)) => value,
        Err(_) => return ApiError::bad("invalid graph request JSON").into_response(),
    };
    // Separate graph fields before decoding the strict completion schema.
    let mut value = value;
    let Some(object) = value.as_object_mut() else {
        return ApiError::bad("graph request must be an object").into_response();
    };
    let graph = match object
        .remove("graph")
        .map(serde_json::from_value::<crate::graph2text::Graph>)
    {
        Some(Ok(graph)) => graph,
        _ => return ApiError::bad("invalid or missing graph").into_response(),
    };
    let options = match object.remove("options") {
        Some(value) => match serde_json::from_value::<crate::graph2text::Options>(value) {
            Ok(options) => options,
            Err(_) => return ApiError::bad("invalid graph options").into_response(),
        },
        None => crate::graph2text::Options::default(),
    };
    let mut request: ApiRequest = match serde_json::from_value(value) {
        Ok(request) => request,
        Err(_) => return ApiError::bad("unsupported graph generation fields").into_response(),
    };
    if request.prompt.is_some() || request.messages.is_some() {
        return ApiError::bad("graph requests cannot include prompt or messages").into_response();
    }
    let prompt = match crate::graph2text::prompt(&graph, &options) {
        Ok(prompt) => prompt,
        Err(error) => return ApiError::bad(error.to_string()).into_response(),
    };
    // Use a configured chat template for instruction models, or a plain completion.
    let chat = state.config.chat_template.is_some();
    if chat {
        request.messages = Some(vec![Message {
            role: "user".into(),
            content: prompt,
        }]);
    } else {
        request.prompt = Some(prompt);
    }
    generate(state, Ok(Json(request)), chat).await
}

fn prepare(
    request: ApiRequest,
    config: &HostConfig,
    chat: bool,
    id: String,
) -> Result<(GenerateRequest, bool, bool), ApiError> {
    if request.model != config.model_name {
        return Err(ApiError::new(
            StatusCode::NOT_FOUND,
            "requested model is not hosted",
        ));
    }
    if request.n.unwrap_or(1) != 1 {
        return Err(ApiError::bad("only n=1 is supported"));
    }
    if request.stream_options.is_some() && !request.stream {
        return Err(ApiError::bad("stream_options requires stream=true"));
    }
    let prompt = if chat {
        if request.prompt.is_some() {
            return Err(ApiError::bad("chat requests use messages, not prompt"));
        }
        render_chat(
            request
                .messages
                .ok_or_else(|| ApiError::bad("messages is required"))?,
            config.chat_template,
        )?
    } else {
        if request.messages.is_some() {
            return Err(ApiError::bad(
                "completion requests use prompt, not messages",
            ));
        }
        request
            .prompt
            .ok_or_else(|| ApiError::bad("prompt must be a string"))?
    };
    if prompt.trim().is_empty() {
        return Err(ApiError::bad("prompt must be nonempty"));
    }
    let max_tokens = request
        .max_tokens
        .unwrap_or(config.max_output_tokens.min(256));
    if max_tokens == 0 || max_tokens > config.max_output_tokens {
        return Err(ApiError::bad(
            "max_tokens exceeds the configured output limit or is zero",
        ));
    }
    let constraint = match request.response_format {
        None => None,
        Some(format) if format.kind == "text" => None,
        Some(format) if format.kind == "json_object" => Some(ConstraintSpec::JsonObject),
        Some(_) => {
            return Err(ApiError::bad(
                "response_format supports only text and json_object",
            ))
        }
    };
    let sampling = SamplingParams {
        max_tokens,
        temperature: request.temperature.unwrap_or(0.7),
        top_p: request.top_p.unwrap_or(0.9),
        top_k: request.top_k,
        min_p: request.min_p.unwrap_or(0.),
        seed: request.seed.unwrap_or(0),
        presence_penalty: request.presence_penalty.unwrap_or(0.),
        frequency_penalty: request.frequency_penalty.unwrap_or(0.),
        repetition_penalty: request.repetition_penalty.unwrap_or(1.),
        stop: match request.stop {
            None => vec![],
            Some(Stop::One(s)) => vec![s],
            Some(Stop::Many(s)) => s,
        },
        ..Default::default()
    };
    if [
        sampling.temperature,
        sampling.top_p,
        sampling.min_p,
        sampling.presence_penalty,
        sampling.frequency_penalty,
        sampling.repetition_penalty,
    ]
    .iter()
    .any(|value| !value.is_finite())
    {
        return Err(ApiError::bad("sampling values must be finite"));
    }
    sampling
        .validate()
        .map_err(|error| ApiError::bad(error.to_string()))?;
    if sampling.top_k == Some(0) {
        return Err(ApiError::bad("top_k must be positive"));
    }
    if sampling.stop.len() > 16 || sampling.stop.iter().any(|stop| stop.len() > 1024) {
        return Err(ApiError::bad(
            "at most 16 stop strings of up to 1024 bytes are supported",
        ));
    }
    Ok((
        GenerateRequest {
            id,
            prompt,
            sampling,
            constraint,
        },
        request.stream,
        request.stream_options.is_some_and(|v| v.include_usage),
    ))
}
fn render_chat(messages: Vec<Message>, template: Option<ChatTemplate>) -> Result<String, ApiError> {
    let template = template.ok_or_else(|| {
        ApiError::bad("chat requires an explicit --chat-template chatml or llama3")
    })?;
    if messages.is_empty() {
        return Err(ApiError::bad("messages must be nonempty"));
    }
    let mut prompt = if matches!(template, ChatTemplate::Llama3) {
        "<|begin_of_text|>".to_owned()
    } else {
        String::new()
    };
    for message in messages {
        if !matches!(message.role.as_str(), "system" | "user" | "assistant") {
            return Err(ApiError::bad(
                "supported roles are system, user and assistant; tool calls are not supported",
            ));
        }
        match template {
            ChatTemplate::ChatMl => prompt.push_str(&format!(
                "<|im_start|>{}\n{}<|im_end|>\n",
                message.role, message.content
            )),
            ChatTemplate::Llama3 => prompt.push_str(&format!(
                "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
                message.role, message.content
            )),
        }
    }
    prompt.push_str(match template {
        ChatTemplate::ChatMl => "<|im_start|>assistant\n",
        ChatTemplate::Llama3 => "<|start_header_id|>assistant<|end_header_id|>\n\n",
    });
    Ok(prompt)
}

async fn generate(
    state: HostState,
    body: Result<Json<ApiRequest>, JsonRejection>,
    chat: bool,
) -> Response {
    let request = match body {
        Ok(Json(request)) => request,
        Err(error) => {
            let status = if error.status() == StatusCode::PAYLOAD_TOO_LARGE {
                StatusCode::PAYLOAD_TOO_LARGE
            } else {
                StatusCode::BAD_REQUEST
            };
            return ApiError::new(status, "invalid JSON body or unsupported request fields")
                .into_response();
        }
    };
    let id = format!(
        "cmpl-{}-{}",
        now(),
        state.ids.fetch_add(1, Ordering::Relaxed)
    );
    let (request, streaming, include_usage) =
        match prepare(request, &state.config, chat, id.clone()) {
            Ok(value) => value,
            Err(error) => return error.into_response(),
        };
    if !state.metrics.ready.load(Ordering::Acquire) {
        return ApiError::new(StatusCode::SERVICE_UNAVAILABLE, "model worker unavailable")
            .into_response();
    }
    if !state.metrics.reserve(state.config.max_pending_requests) {
        return ApiError::new(StatusCode::TOO_MANY_REQUESTS, "model request queue is full")
            .into_response();
    }
    let (acknowledge, accepted) = oneshot::channel();
    let (stream_sender, stream_receiver) = mpsc::channel(64);
    let (result_sender, result_receiver) = oneshot::channel();
    let sink = if streaming {
        Sink::Streaming(stream_sender)
    } else {
        Sink::Buffered(result_sender)
    };
    let command = Command {
        request,
        sink,
        acknowledge,
        chat,
        include_usage,
        created: now(),
        deadline: Instant::now() + state.config.request_timeout,
    };
    if state.commands.try_send(command).is_err() {
        state.metrics.release();
        return ApiError::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "model worker unavailable or busy",
        )
        .into_response();
    }
    match accepted.await {
        Ok(Ok(())) => {}
        Ok(Err(error)) => return error.into_response(),
        Err(_) => {
            return ApiError::new(StatusCode::SERVICE_UNAVAILABLE, "model worker stopped")
                .into_response()
        }
    }
    let mut response = if streaming {
        Sse::new(ReceiverStream::new(stream_receiver))
            .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
            .into_response()
    } else {
        match result_receiver.await {
            Ok(Ok(output)) => {
                Json(completion(&output, &state.config.model_name, now(), chat)).into_response()
            }
            Ok(Err(error)) => error.into_response(),
            Err(_) => ApiError::new(StatusCode::SERVICE_UNAVAILABLE, "model worker stopped")
                .into_response(),
        }
    };
    if let Ok(value) = id.parse() {
        response.headers_mut().insert("x-request-id", value);
    }
    response
}
fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
fn usage(output: &GenerateResponse) -> Value {
    json!({"prompt_tokens":output.prompt_tokens,"completion_tokens":output.token_ids.len(),"total_tokens":output.prompt_tokens + output.token_ids.len()})
}
fn reason(output: &GenerateResponse) -> &'static str {
    if output.finish_reason == FinishReason::Length {
        "length"
    } else {
        "stop"
    }
}
fn completion(output: &GenerateResponse, model: &str, created: u64, chat: bool) -> Value {
    let choice = if chat {
        json!({"index":0,"message":{"role":"assistant","content":output.text},"finish_reason":reason(output),"logprobs":null})
    } else {
        json!({"index":0,"text":output.text,"finish_reason":reason(output),"logprobs":null})
    };
    json!({"id":output.id,"object":if chat {"chat.completion"} else {"text_completion"},"created":created,"model":model,"choices":[choice],"usage":usage(output)})
}

struct Command {
    request: GenerateRequest,
    sink: Sink,
    acknowledge: oneshot::Sender<Result<(), ApiError>>,
    chat: bool,
    include_usage: bool,
    created: u64,
    deadline: Instant,
}
enum Sink {
    Buffered(oneshot::Sender<Result<GenerateResponse, ApiError>>),
    Streaming(mpsc::Sender<SseItem>),
}
impl Sink {
    fn closed(&self) -> bool {
        match self {
            Self::Buffered(sender) => sender.is_closed(),
            Self::Streaming(sender) => sender.is_closed(),
        }
    }
    fn event(&self, value: Value) -> bool {
        match self {
            Self::Streaming(sender) => sender
                .try_send(Ok(Event::default().data(value.to_string())))
                .is_ok(),
            _ => true,
        }
    }
}
struct Pending {
    sink: Sink,
    chat: bool,
    include_usage: bool,
    created: u64,
    deadline: Instant,
    previous: String,
    sent: String,
    stops: Vec<String>,
}
impl Pending {
    fn chunk(&self, id: &str, model: &str, text: &str, finish: Option<&str>, role: bool) -> Value {
        let choice = if self.chat {
            let mut delta = json!({});
            if !text.is_empty() {
                delta["content"] = json!(text);
            }
            if role {
                delta["role"] = json!("assistant");
            }
            json!({"index":0,"delta":delta,"finish_reason":finish,"logprobs":null})
        } else {
            json!({"index":0,"text":text,"finish_reason":finish,"logprobs":null})
        };
        json!({"id":id,"object":if self.chat {"chat.completion.chunk"} else {"text_completion"},"created":self.created,"model":model,"choices":[choice]})
    }
    fn progress(&mut self, id: &str, model: &str, current: String) -> bool {
        if matches!(self.sink, Sink::Buffered(_)) {
            return true;
        }
        // Only emit the prefix stable across two decode steps. Buffer incomplete
        // UTF-8 replacement suffixes and any potential stop string prefix.
        let stable = common_prefix(&self.previous, &current);
        let stable = stable.split('\u{fffd}').next().unwrap_or("");
        let stable = without_stop_prefix(stable, &self.stops);
        if !current.starts_with(&self.sent) {
            return false;
        }
        if stable.starts_with(&self.sent) && stable.len() > self.sent.len() {
            let delta = &stable[self.sent.len()..];
            if !self.sink.event(self.chunk(id, model, delta, None, false)) {
                return false;
            }
            self.sent = stable.to_owned();
        }
        self.previous = current;
        true
    }
    fn finish(self, output: GenerateResponse, model: &str) {
        if matches!(self.sink, Sink::Buffered(_)) {
            if let Sink::Buffered(sender) = self.sink {
                let _ = sender.send(Ok(output));
            }
            return;
        }
        if !output.text.starts_with(&self.sent) {
            self.fail(ApiError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "backend revised already streamed text",
            ));
            return;
        }
        if !self.sink.event(self.chunk(
            &output.id,
            model,
            &output.text[self.sent.len()..],
            None,
            false,
        )) {
            return;
        }
        if !self
            .sink
            .event(self.chunk(&output.id, model, "", Some(reason(&output)), false))
        {
            return;
        }
        if self.include_usage {
            self.sink.event(json!({"id":output.id,"object":if self.chat {"chat.completion.chunk"} else {"text_completion"},"created":self.created,"model":model,"choices":[],"usage":usage(&output)}));
        }
        if let Sink::Streaming(sender) = self.sink {
            let _ = sender.try_send(Ok(Event::default().data("[DONE]")));
        }
    }
    fn fail(self, error: ApiError) {
        match self.sink {
            Sink::Buffered(sender) => {
                let _ = sender.send(Err(error));
            }
            Sink::Streaming(sender) => {
                let _ = sender.try_send(Ok(Event::default().data(error.value().to_string())));
            }
        }
    }
}
fn common_prefix<'a>(a: &'a str, b: &str) -> &'a str {
    let end = a
        .char_indices()
        .zip(b.chars())
        .take_while(|((_, a), b)| a == b)
        .map(|((at, ch), _)| at + ch.len_utf8())
        .last()
        .unwrap_or(0);
    &a[..end]
}
fn without_stop_prefix<'a>(text: &'a str, stops: &[String]) -> &'a str {
    let mut end = text.len();
    for stop in stops {
        for (start, _) in text.char_indices() {
            if stop.starts_with(&text[start..]) {
                end = end.min(start);
                break;
            }
        }
    }
    &text[..end]
}

fn worker<B: NativeBackend>(
    mut engine: NativeEngine<B>,
    mut receiver: mpsc::Receiver<Command>,
    config: Arc<HostConfig>,
    metrics: Arc<Metrics>,
) {
    let _ready = WorkerReady(metrics.clone());
    let mut pending: HashMap<String, Pending> = HashMap::new();
    loop {
        if engine.is_idle() {
            let Some(command) = receiver.blocking_recv() else {
                break;
            };
            admit(command, &mut engine, &mut pending, &config, &metrics);
        }
        // Limit admission per iteration so a busy producer cannot starve decoding.
        for _ in 0..config.max_pending_requests {
            match receiver.try_recv() {
                Ok(command) => admit(command, &mut engine, &mut pending, &config, &metrics),
                Err(_) => break,
            }
        }
        let expired: Vec<_> = pending
            .iter()
            .filter(|(_, p)| p.sink.closed() || Instant::now() >= p.deadline)
            .map(|(id, _)| id.clone())
            .collect();
        for id in expired {
            engine.cancel(&id);
            if let Some(request) = pending.remove(&id) {
                request.fail(ApiError::new(
                    StatusCode::GATEWAY_TIMEOUT,
                    "request timed out or client disconnected",
                ));
                metrics.release();
                metrics.failed.fetch_add(1, Ordering::Relaxed);
            }
        }
        if !engine.is_idle() {
            let result = engine
                .step()
                .and_then(|completed| engine.progress().map(|progress| (completed, progress)));
            match result {
                Ok((completed, progress)) => {
                    for output in completed {
                        if let Some(request) = pending.remove(&output.id) {
                            if Instant::now() >= request.deadline {
                                request.fail(ApiError::new(
                                    StatusCode::GATEWAY_TIMEOUT,
                                    "request timed out",
                                ));
                                metrics.failed.fetch_add(1, Ordering::Relaxed);
                            } else {
                                request.finish(output, &config.model_name);
                                metrics.completed.fetch_add(1, Ordering::Relaxed);
                            }
                            metrics.release();
                        }
                    }
                    for (id, text) in progress {
                        if let Some(request) = pending.get_mut(&id) {
                            if !request.progress(&id, &config.model_name, text) {
                                engine.cancel(&id);
                                if let Some(request) = pending.remove(&id) {
                                    request.fail(ApiError::new(
                                        StatusCode::INTERNAL_SERVER_ERROR,
                                        "stream disconnected, too slow, or backend revised text",
                                    ));
                                    metrics.release();
                                    metrics.failed.fetch_add(1, Ordering::Relaxed);
                                }
                            }
                        }
                    }
                }
                Err(error) => {
                    eprintln!("model worker failed: {error}");
                    engine.abort_all();
                    for (_, request) in pending.drain() {
                        request.fail(ApiError::new(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "model inference failed",
                        ));
                        metrics.release();
                        metrics.failed.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }
        metrics.active.store(engine.active(), Ordering::Relaxed);
        metrics.queued.store(engine.queued(), Ordering::Relaxed);
    }
    metrics.ready.store(false, Ordering::Release);
    metrics.active.store(0, Ordering::Relaxed);
    metrics.queued.store(0, Ordering::Relaxed);
}
fn admit<B: NativeBackend>(
    command: Command,
    engine: &mut NativeEngine<B>,
    pending: &mut HashMap<String, Pending>,
    config: &HostConfig,
    metrics: &Metrics,
) {
    let Command {
        request,
        sink,
        acknowledge,
        chat,
        include_usage,
        created,
        deadline,
    } = command;
    let checked = engine
        .backend()
        .encode(&request.prompt)
        .map_err(|_| ApiError::bad("prompt could not be tokenized"))
        .and_then(|tokens| {
            if tokens.is_empty() || tokens.len() > config.max_input_tokens {
                return Err(ApiError::bad(
                    "prompt is empty after tokenization or exceeds max_input_tokens",
                ));
            }
            if config.max_context_tokens.is_some_and(|limit| {
                tokens.len().saturating_add(request.sampling.max_tokens) > limit
            }) {
                return Err(ApiError::bad(
                    "prompt plus max_tokens exceeds model context capacity",
                ));
            }
            if Instant::now() >= deadline {
                return Err(ApiError::new(
                    StatusCode::GATEWAY_TIMEOUT,
                    "request expired in queue",
                ));
            }
            Ok(())
        });
    if let Err(error) = checked {
        let _ = acknowledge.send(Err(error));
        metrics.release();
        metrics.failed.fetch_add(1, Ordering::Relaxed);
        return;
    }
    let id = request.id.clone();
    let stops = request.sampling.stop.clone();
    if let Err(error) = engine.submit(request) {
        let _ = acknowledge.send(Err(ApiError::bad(error.to_string())));
        metrics.release();
        metrics.failed.fetch_add(1, Ordering::Relaxed);
        return;
    }
    let request = Pending {
        sink,
        chat,
        include_usage,
        created,
        deadline,
        previous: String::new(),
        sent: String::new(),
        stops,
    };
    if acknowledge.send(Ok(())).is_err() || request.sink.closed() {
        engine.cancel(&id);
        metrics.release();
        return;
    }
    if chat {
        request
            .sink
            .event(request.chunk(&id, &config.model_name, "", None, true));
    }
    pending.insert(id, request);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::NativeResult;
    use axum::body::{to_bytes, Body};
    use std::sync::Mutex;
    use tower::ServiceExt;

    struct Toy {
        delay: Duration,
        fail_once: bool,
        removed: Arc<Mutex<Vec<String>>>,
    }
    impl NativeBackend for Toy {
        fn encode(&self, prompt: &str) -> NativeResult<Vec<u32>> {
            Ok(vec![0; prompt.split_whitespace().count()])
        }
        fn decode(&self, tokens: &[u32]) -> NativeResult<String> {
            Ok(tokens
                .iter()
                .filter_map(|id| match id {
                    0 => Some('a'),
                    1 => Some('b'),
                    _ => None,
                })
                .collect())
        }
        fn logits(&mut self, _: &str, _: &[u32], position: usize) -> NativeResult<Vec<f32>> {
            if self.fail_once {
                self.fail_once = false;
                return Err(NativeError("intentional batch failure".into()));
            }
            if !self.delay.is_zero() {
                std::thread::sleep(self.delay);
            }
            Ok(if position < 4 {
                vec![10., 0., -10.]
            } else {
                vec![-10., -10., 10.]
            })
        }
        fn remove_sequence(&mut self, id: &str) {
            self.removed.lock().unwrap().push(id.into());
        }
    }
    fn app(
        config: HostConfig,
        delay: Duration,
        fail_once: bool,
    ) -> (Router, Arc<Mutex<Vec<String>>>) {
        let removed = Arc::new(Mutex::new(vec![]));
        (
            router(
                Toy {
                    delay,
                    fail_once,
                    removed: removed.clone(),
                },
                vec![2],
                config,
            )
            .unwrap(),
            removed,
        )
    }
    fn payload() -> Value {
        json!({"model":"lighter","prompt":"hello","max_tokens":8,"temperature":0.0})
    }
    async fn call(app: Router, path: &str, body: Option<Value>, key: Option<&str>) -> Response {
        let mut request = axum::http::Request::builder().uri(path);
        if body.is_some() {
            request = request
                .method("POST")
                .header("content-type", "application/json");
        }
        if let Some(key) = key {
            request = request.header("authorization", format!("Bearer {key}"));
        }
        app.oneshot(
            request
                .body(
                    body.map(|b| Body::from(b.to_string()))
                        .unwrap_or_else(Body::empty),
                )
                .unwrap(),
        )
        .await
        .unwrap()
    }
    async fn json_response(response: Response) -> Value {
        serde_json::from_slice(&to_bytes(response.into_body(), 1024 * 1024).await.unwrap()).unwrap()
    }
    #[tokio::test]
    async fn model_discovery_and_completion_usage() {
        let (app, removed) = app(HostConfig::default(), Duration::ZERO, false);
        let models = json_response(call(app.clone(), "/v1/models", None, None).await).await;
        assert_eq!(models["data"][0]["id"], "lighter");
        let response = call(app.clone(), "/v1/completions", Some(payload()), None).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert!(response.headers().contains_key("x-request-id"));
        let response = json_response(response).await;
        assert_eq!(response["choices"][0]["text"], "aaa");
        assert_eq!(response["choices"][0]["finish_reason"], "stop");
        assert_eq!(response["usage"]["prompt_tokens"], 1);
        assert_eq!(response["usage"]["completion_tokens"], 4);
        assert_eq!(removed.lock().unwrap().len(), 1);
    }
    #[tokio::test]
    async fn bearer_auth_health_and_errors() {
        let (app, _) = app(
            HostConfig {
                api_key: Some("test-key".into()),
                ..Default::default()
            },
            Duration::ZERO,
            false,
        );
        assert_eq!(
            call(app.clone(), "/health", None, None).await.status(),
            StatusCode::OK
        );
        let denied = call(app.clone(), "/v1/models", None, None).await;
        assert_eq!(denied.status(), StatusCode::UNAUTHORIZED);
        assert!(json_response(denied).await["error"]["message"].is_string());
        assert_eq!(
            call(app, "/v1/models", None, Some("test-key"))
                .await
                .status(),
            StatusCode::OK
        );
    }
    #[tokio::test]
    async fn unsupported_fields_and_limits_are_rejected_before_inference() {
        let (app, removed) = app(
            HostConfig {
                max_input_tokens: 2,
                max_context_tokens: Some(9),
                max_output_tokens: 8,
                ..Default::default()
            },
            Duration::ZERO,
            false,
        );
        for request in [
            json!({"model":"lighter","prompt":["hello"]}),
            json!({"model":"lighter","prompt":"hello","tools":[]}),
            json!({"model":"lighter","prompt":"hello","n":2}),
            json!({"model":"lighter","prompt":"hello","max_tokens":0}),
            json!({"model":"lighter","prompt":"hello","max_tokens":9}),
            json!({"model":"lighter","prompt":"hello","temperature":-1}),
            json!({"model":"lighter","prompt":"hello","temperature":1e100}),
            json!({"model":"lighter","prompt":"hello","stream_options":{"include_usage":true}}),
            json!({"model":"lighter","prompt":"hello","response_format":{"type":"json_schema"}}),
            json!({"model":"lighter","prompt":"hello world again","max_tokens":1}),
            json!({"model":"lighter","prompt":"hello world","max_tokens":8}),
        ] {
            assert_eq!(
                call(app.clone(), "/v1/completions", Some(request), None)
                    .await
                    .status(),
                StatusCode::BAD_REQUEST
            );
        }
        let mut wrong = payload();
        wrong["model"] = json!("other");
        assert_eq!(
            call(app, "/v1/completions", Some(wrong), None)
                .await
                .status(),
            StatusCode::NOT_FOUND
        );
        assert!(removed.lock().unwrap().is_empty());
    }
    #[tokio::test]
    async fn chat_format_and_assistant_response() {
        let (app, _) = app(
            HostConfig {
                chat_template: Some(ChatTemplate::ChatMl),
                ..Default::default()
            },
            Duration::ZERO,
            false,
        );
        let response = call(app, "/v1/chat/completions", Some(json!({"model":"lighter","messages":[{"role":"user","content":"hello"}],"max_tokens":1,"temperature":0.0})), None).await;
        assert_eq!(response.status(), StatusCode::OK);
        let response = json_response(response).await;
        assert_eq!(response["object"], "chat.completion");
        assert_eq!(response["choices"][0]["message"]["role"], "assistant");
        assert_eq!(response["choices"][0]["finish_reason"], "length");
        let prompt = render_chat(
            vec![Message {
                role: "user".into(),
                content: "hello".into(),
            }],
            Some(ChatTemplate::Llama3),
        )
        .unwrap();
        assert_eq!(prompt, "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nhello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n");
    }
    #[tokio::test]
    async fn disabled_chat_and_multimodal_content_are_rejected() {
        let (app, _) = app(HostConfig::default(), Duration::ZERO, false);
        for request in [
            json!({"model":"lighter","messages":[{"role":"user","content":"hi"}]}),
            json!({"model":"lighter","messages":[{"role":"user","content":[]}]}),
            json!({"model":"lighter","messages":[{"role":"tool","content":"hi"}]}),
        ] {
            assert_eq!(
                call(app.clone(), "/v1/chat/completions", Some(request), None)
                    .await
                    .status(),
                StatusCode::BAD_REQUEST
            );
        }
    }
    #[tokio::test]
    async fn streaming_chunks_reconstruct_text_and_exclude_stop_prefixes() {
        let (app, _) = app(HostConfig::default(), Duration::from_millis(10), false);
        let mut request = payload();
        request["stream"] = json!(true);
        request["stream_options"] = json!({"include_usage":true});
        request["stop"] = json!("aa");
        let response = call(app, "/v1/completions", Some(request), None).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert!(response.headers()["content-type"]
            .to_str()
            .unwrap()
            .starts_with("text/event-stream"));
        let body = String::from_utf8(
            to_bytes(response.into_body(), 1024 * 1024)
                .await
                .unwrap()
                .to_vec(),
        )
        .unwrap();
        let mut text = String::new();
        let mut usage = false;
        let mut finished = false;
        for line in body.lines().filter_map(|l| l.strip_prefix("data: ")) {
            if line == "[DONE]" {
                finished = true;
                continue;
            }
            let data: Value = serde_json::from_str(line).unwrap();
            assert!(data.get("error").is_none());
            if let Some(part) = data["choices"][0]["text"].as_str() {
                text.push_str(part);
            }
            usage |= data.get("usage").is_some();
        }
        assert_eq!(text, "");
        assert!(usage && finished);
    }
    #[tokio::test]
    async fn streaming_is_incremental_before_completion() {
        let (app, _) = app(HostConfig::default(), Duration::from_millis(40), false);
        let mut request = payload();
        request["stream"] = json!(true);
        let response = call(app, "/v1/completions", Some(request), None).await;
        let mut body = response.into_body();
        use axum::body::HttpBody;
        use std::future::poll_fn;
        let first = poll_fn(|cx| std::pin::Pin::new(&mut body).poll_frame(cx))
            .await
            .unwrap()
            .unwrap()
            .into_data()
            .unwrap();
        let first = String::from_utf8(first.to_vec()).unwrap();
        assert!(first.contains("\"text\":\"a\""));
        assert!(!first.contains("[DONE]"));
        let rest = String::from_utf8(to_bytes(body, 1024 * 1024).await.unwrap().to_vec()).unwrap();
        assert!(rest.contains("[DONE]"));
    }
    #[tokio::test]
    async fn client_disconnect_releases_backend_cache() {
        let (app, removed) = app(HostConfig::default(), Duration::from_millis(10), false);
        let mut request = payload();
        request["stream"] = json!(true);
        let response = call(app.clone(), "/v1/completions", Some(request), None).await;
        drop(response);
        tokio::time::timeout(Duration::from_secs(2), async {
            while removed.lock().unwrap().is_empty() {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        assert_eq!(
            call(app, "/health", None, None).await.status(),
            StatusCode::OK
        );
    }
    #[tokio::test]
    async fn bounded_queue_and_timeout() {
        let (app, _) = app(
            HostConfig {
                max_pending_requests: 1,
                request_timeout: Duration::from_millis(20),
                ..Default::default()
            },
            Duration::from_millis(100),
            false,
        );
        let mut request = payload();
        request["stream"] = json!(true);
        let held = call(app.clone(), "/v1/completions", Some(request), None).await;
        assert_eq!(
            call(app.clone(), "/v1/completions", Some(payload()), None)
                .await
                .status(),
            StatusCode::TOO_MANY_REQUESTS
        );
        let body = String::from_utf8(
            to_bytes(held.into_body(), 1024 * 1024)
                .await
                .unwrap()
                .to_vec(),
        )
        .unwrap();
        assert!(body.contains("timed out"));
        let response = call(app, "/v1/completions", Some(payload()), None).await;
        assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
    }
    #[tokio::test]
    async fn batch_failure_cleans_caches_and_next_request_recovers() {
        let (app, removed) = app(HostConfig::default(), Duration::ZERO, true);
        assert_eq!(
            call(app.clone(), "/v1/completions", Some(payload()), None)
                .await
                .status(),
            StatusCode::INTERNAL_SERVER_ERROR
        );
        assert_eq!(
            call(app.clone(), "/v1/completions", Some(payload()), None)
                .await
                .status(),
            StatusCode::OK
        );
        assert_eq!(removed.lock().unwrap().len(), 2);
        assert_eq!(
            call(app, "/health", None, None).await.status(),
            StatusCode::OK
        );
    }
    #[test]
    fn unicode_and_stop_buffering() {
        assert_eq!(common_prefix("héa", "héb"), "hé");
        assert_eq!(without_stop_prefix("hello ST", &["STOP".into()]), "hello ");
        assert_eq!(without_stop_prefix("hé", &["éx".into()]), "h");
        assert!(HostConfig {
            batch_size: 0,
            ..Default::default()
        }
        .validate()
        .is_err());
    }
    #[tokio::test]
    async fn overlapping_requests_share_a_decode_batch() {
        struct Batched {
            toy: Toy,
            largest: Arc<AtomicUsize>,
        }
        impl NativeBackend for Batched {
            fn encode(&self, text: &str) -> NativeResult<Vec<u32>> {
                self.toy.encode(text)
            }
            fn decode(&self, ids: &[u32]) -> NativeResult<String> {
                self.toy.decode(ids)
            }
            fn logits(&mut self, id: &str, ids: &[u32], position: usize) -> NativeResult<Vec<f32>> {
                self.toy.logits(id, ids, position)
            }
            fn logits_batch(
                &mut self,
                batch: &[crate::native::BackendInput<'_>],
            ) -> NativeResult<Vec<Vec<f32>>> {
                self.largest.fetch_max(batch.len(), Ordering::Relaxed);
                batch
                    .iter()
                    .map(|input| {
                        self.toy
                            .logits(input.sequence_id, input.tokens, input.position)
                    })
                    .collect()
            }
            fn remove_sequence(&mut self, id: &str) {
                self.toy.remove_sequence(id);
            }
        }
        let largest = Arc::new(AtomicUsize::new(0));
        let backend = Batched {
            toy: Toy {
                delay: Duration::from_millis(10),
                fail_once: false,
                removed: Arc::new(Mutex::new(vec![])),
            },
            largest: largest.clone(),
        };
        let app = router(backend, vec![2], HostConfig::default()).unwrap();
        let (first, second) = tokio::join!(
            call(app.clone(), "/v1/completions", Some(payload()), None),
            call(app, "/v1/completions", Some(payload()), None)
        );
        assert_eq!(first.status(), StatusCode::OK);
        assert_eq!(second.status(), StatusCode::OK);
        assert_eq!(largest.load(Ordering::Relaxed), 2);
    }
    #[test]
    fn json_object_root_is_constrained() {
        let constraint = ConstraintSpec::JsonObject.compile().unwrap();
        assert!(constraint.allows_prefix(" {"));
        assert!(!constraint.allows_prefix("["));
        assert!(!constraint.is_complete("123"));
        assert!(constraint.is_complete(r#"{"ok":true}"#));
        struct ObjectBackend;
        impl NativeBackend for ObjectBackend {
            fn encode(&self, _: &str) -> NativeResult<Vec<u32>> {
                Ok(vec![3])
            }
            fn decode(&self, ids: &[u32]) -> NativeResult<String> {
                Ok(ids
                    .iter()
                    .filter_map(|id| match id {
                        0 => Some('{'),
                        1 => Some('}'),
                        2 => None,
                        _ => Some('a'),
                    })
                    .collect())
            }
            fn logits(&mut self, _: &str, _: &[u32], position: usize) -> NativeResult<Vec<f32>> {
                // EOS wins unless the engine explicitly masks it for object mode.
                Ok(if position == 1 {
                    vec![8., -10., 10., -10.]
                } else {
                    vec![-10., 8., 10., -10.]
                })
            }
        }
        let mut engine = NativeEngine::new(ObjectBackend, 1, vec![2]).unwrap();
        engine
            .submit(GenerateRequest {
                id: "json".into(),
                prompt: "json".into(),
                sampling: SamplingParams {
                    max_tokens: 4,
                    temperature: 0.,
                    ..Default::default()
                },
                constraint: Some(ConstraintSpec::JsonObject),
            })
            .unwrap();
        assert_eq!(engine.run_to_completion().unwrap()[0].text, "{}");
    }
}

#[cfg(test)]
mod graph_api_tests {
    use super::*;
    use crate::native::NativeResult;
    use axum::{
        body::{to_bytes, Body},
        http::Request,
    };
    use tower::ServiceExt;
    struct Backend;
    impl NativeBackend for Backend {
        fn encode(&self, prompt: &str) -> NativeResult<Vec<u32>> {
            assert!(prompt.contains("FACTS_JSON:") && prompt.contains("Alice"));
            Ok(vec![0])
        }
        fn decode(&self, tokens: &[u32]) -> NativeResult<String> {
            Ok(tokens
                .iter()
                .filter(|&&x| x == 1)
                .map(|_| "graph answer")
                .collect())
        }
        fn logits(&mut self, _: &str, _: &[u32], position: usize) -> NativeResult<Vec<f32>> {
            Ok(if position == 1 {
                vec![-100.0, 100.0, -100.0]
            } else {
                vec![-100.0, -100.0, 100.0]
            })
        }
    }
    #[tokio::test]
    async fn graph_endpoint_validates_and_generates() {
        let app = router(Backend, vec![2], HostConfig::default()).unwrap();
        let graph = json!({"nodes":[{"id":"a","label":"Alice"}],"edges":[]});
        let request = |value: Value| {
            Request::builder()
                .method("POST")
                .uri("/v1/graph/completions")
                .header("content-type", "application/json")
                .body(Body::from(value.to_string()))
                .unwrap()
        };
        let response = app
            .clone()
            .oneshot(request(
                json!({"model":"lighter","graph":graph,"temperature":0,"max_tokens":4}),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 10000).await.unwrap()).unwrap();
        assert_eq!(body["choices"][0]["text"], "graph answer");
        for bad in [
            json!({"model":"lighter","graph":{"nodes":[],"edges":[]}}),
            json!({"model":"lighter","graph":graph,"prompt":"override"}),
            json!({"model":"lighter","graph":graph,"unknown":true}),
            json!({"model":"lighter","graph":graph,"options":{"root":"absent"}}),
        ] {
            assert_eq!(
                app.clone().oneshot(request(bad)).await.unwrap().status(),
                StatusCode::BAD_REQUEST
            );
        }
    }
    #[tokio::test]
    async fn graph_endpoint_chat_stream_uses_shared_host_protocol() {
        let app = router(
            Backend,
            vec![2],
            HostConfig {
                chat_template: Some(ChatTemplate::Llama3),
                ..HostConfig::default()
            },
        )
        .unwrap();
        let body = json!({"model":"lighter", "graph":{"nodes":[{"id":"a","label":"Alice"}],"edges":[]},"stream":true,"stream_options":{"include_usage":true},"temperature":0,"max_tokens":4});
        let request = Request::builder()
            .method("POST")
            .uri("/v1/graph/completions")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()["content-type"], "text/event-stream");
        let body = String::from_utf8(
            to_bytes(response.into_body(), 10000)
                .await
                .unwrap()
                .to_vec(),
        )
        .unwrap();
        assert!(body.contains("graph answer"));
        assert!(body.contains("assistant"));
        assert!(body.contains("prompt_tokens"));
        assert!(body.contains("[DONE]"));
    }
}
