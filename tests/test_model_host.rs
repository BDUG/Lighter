//! Real TCP/HTTP integration with a tiny local safetensors decoder checkpoint.
#![cfg(feature = "server")]
use candlelighter::model_host::{router, ChatTemplate, HostConfig};
use candlelighter::native::{HuggingFaceArtifacts, HuggingFaceBackend};
use candlelighter::native_advanced::{QuantizationConfig, QuantizationType};
use candlelighter::NativeLlama;
use safetensors::{tensor::TensorView, Dtype};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs,
    path::PathBuf,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "lighter-host-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&path).unwrap();
        let result = Self(path);
        fs::write(
            result.0.join("config.json"),
            json!({
                "model_type":"llama", "vocab_size":4,"hidden_size":2,"intermediate_size":2,
                "num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,
                "max_position_embeddings":32,"rms_norm_eps":1e-6,"eos_token_id":2
            })
            .to_string(),
        )
        .unwrap();
        fs::write(result.0.join("tokenizer.json"), json!({
            "version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,
            "pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,
            "model":{"type":"WordLevel","vocab":{"<unk>":0,"hello":1,"<eos>":2,"other":3},"unk_token":"<unk>"}
        }).to_string()).unwrap();
        let mut tensors: BTreeMap<String, (Vec<usize>, Vec<u8>)> = BTreeMap::new();
        let mut add = |name: &str, shape: Vec<usize>, values: Vec<f32>| {
            tensors.insert(
                name.into(),
                (
                    shape,
                    values.into_iter().flat_map(f32::to_le_bytes).collect(),
                ),
            );
        };
        add(
            "model.embed_tokens.weight",
            vec![4, 2],
            vec![1., 0., 1., 0., 1., 0., 1., 0.],
        );
        add(
            "lm_head.weight",
            vec![4, 2],
            vec![0., 0., 1., 0., 0., 0., -1., 0.],
        );
        for name in [
            "model.norm.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
        ] {
            add(name, vec![2], vec![1., 1.]);
        }
        for name in [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ] {
            add(
                &format!("model.layers.0.{name}.weight"),
                vec![2, 2],
                vec![0.; 4],
            );
        }
        let views: Vec<_> = tensors
            .iter()
            .map(|(name, (shape, bytes))| {
                (
                    name.as_str(),
                    TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
                )
            })
            .collect();
        safetensors::serialize_to_file(views, &None, &result.0.join("model.safetensors")).unwrap();
        result
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn real_checkpoint_tcp_completion_chat_and_streaming() {
    let fixture = Fixture::new();
    let artifacts = HuggingFaceArtifacts::from_dir(&fixture.0).unwrap();
    let model = NativeLlama::load(&artifacts).unwrap();
    let backend = HuggingFaceBackend::new(model, &artifacts).unwrap();
    let app = router(
        backend,
        artifacts.config.eos_token_ids(),
        HostConfig {
            model_name: "tiny-test".into(),
            chat_template: Some(ChatTemplate::ChatMl),
            max_context_tokens: Some(32),
            api_key: Some("fixture-key".into()),
            ..Default::default()
        },
    )
    .unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let result = tokio::task::spawn_blocking(move || {
        let client = ureq::AgentBuilder::new().timeout(Duration::from_secs(5)).build();
        assert_eq!(client.get(&format!("{base}/health")).call().unwrap().status(), 200);
        assert!(matches!(client.get(&format!("{base}/v1/models")).call(), Err(ureq::Error::Status(401, _))));
        let models: Value = client.get(&format!("{base}/v1/models")).set("Authorization", "Bearer fixture-key").call().unwrap().into_json().unwrap();
        assert_eq!(models["data"][0]["id"], "tiny-test");
        let request = json!({"model":"tiny-test","prompt":"hello","max_tokens":3,"temperature":0.0});
        let response: Value = client.post(&format!("{base}/v1/completions")).set("Authorization", "Bearer fixture-key").send_json(request.clone()).unwrap().into_json().unwrap();
        assert_eq!(response["choices"][0]["text"], "hello hello hello");
        assert_eq!(response["choices"][0]["finish_reason"], "length");
        assert_eq!(response["usage"]["total_tokens"], 4);
        let mut stream = request; stream["stream"] = json!(true);
        let body = client.post(&format!("{base}/v1/completions")).set("Authorization", "Bearer fixture-key").send_json(stream).unwrap().into_string().unwrap();
        assert!(body.contains("data: [DONE]"));
        let combined: String = body.lines().filter_map(|line| line.strip_prefix("data: ")).filter(|data| *data != "[DONE]").map(|data| {
            let value: Value = serde_json::from_str(data).unwrap();
            assert!(value.get("error").is_none());
            value["choices"][0]["text"].as_str().unwrap_or("").to_owned()
        }).collect();
        assert_eq!(combined, "hello hello hello");
        let chat: Value = client.post(&format!("{base}/v1/chat/completions")).set("Authorization", "Bearer fixture-key").send_json(json!({
            "model":"tiny-test","messages":[{"role":"user","content":"hello"}],"temperature":0.0,"max_tokens":2
        })).unwrap().into_json().unwrap();
        assert_eq!(chat["choices"][0]["message"]["content"], "hello hello");
        assert_eq!(chat["choices"][0]["message"]["role"], "assistant");
    }).await;
    task.abort();
    result.unwrap();
}

#[tokio::test]
async fn both_quantized_loaders_are_servable() {
    let fixture = Fixture::new();
    for dtype in [QuantizationType::Int8, QuantizationType::Int4] {
        let artifacts = HuggingFaceArtifacts::from_dir(&fixture.0).unwrap();
        let model = NativeLlama::load_quantized(
            &artifacts,
            QuantizationConfig {
                dtype,
                group_size: 64,
            },
        )
        .unwrap();
        let backend = HuggingFaceBackend::new(model, &artifacts).unwrap();
        let app = router(backend, vec![2], HostConfig::default()).unwrap();
        use tower::ServiceExt;
        let response = app.oneshot(axum::http::Request::builder().uri("/v1/completions").method("POST").header("content-type", "application/json")
            .body(axum::body::Body::from(json!({"model":"lighter","prompt":"hello","max_tokens":2,"temperature":0.0}).to_string())).unwrap()).await.unwrap();
        assert_eq!(response.status(), 200);
        let body = axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let data: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(data["choices"][0]["text"], "hello hello");
    }
}

#[test]
fn explicit_bos_chat_prompt_does_not_double_tokenizer_bos() {
    use candlelighter::native::{NativeBackend, NativeModel, NativeResult};
    let fixture = Fixture::new();
    let config_path = fixture.0.join("config.json");
    let mut config: Value =
        serde_json::from_str(&fs::read_to_string(&config_path).unwrap()).unwrap();
    config["bos_token_id"] = json!(4);
    fs::write(config_path, config.to_string()).unwrap();
    let path = fixture.0.join("tokenizer.json");
    let mut tokenizer: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
    tokenizer["model"]["vocab"]["<bos>"] = json!(4);
    tokenizer["added_tokens"] = json!([{"id":4,"content":"<bos>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}]);
    tokenizer["post_processor"] = json!({
        "type":"TemplateProcessing",
        "single":[{"SpecialToken":{"id":"<bos>","type_id":0}},{"Sequence":{"id":"A","type_id":0}}],
        "pair":[{"SpecialToken":{"id":"<bos>","type_id":0}},{"Sequence":{"id":"A","type_id":0}},{"Sequence":{"id":"B","type_id":1}}],
        "special_tokens":{"<bos>":{"id":"<bos>","ids":[4],"tokens":["<bos>"]}}
    });
    fs::write(path, tokenizer.to_string()).unwrap();
    struct TokenizerOnly;
    impl NativeModel for TokenizerOnly {
        fn forward(&mut self, _: &str, _: &[u32], _: usize) -> NativeResult<Vec<f32>> {
            unreachable!("this test only exercises tokenization")
        }
    }
    let artifacts = HuggingFaceArtifacts::from_dir(&fixture.0).unwrap();
    let backend = HuggingFaceBackend::new(TokenizerOnly, &artifacts).unwrap();
    assert_eq!(backend.encode("hello").unwrap(), vec![4, 1]);
    assert_eq!(backend.encode("<bos>hello").unwrap(), vec![4, 1]);
}
