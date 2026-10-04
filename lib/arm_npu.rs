//! Thread-affine TensorFlow Lite C runtime with an Arm NN-compatible external delegate.
//! Models must accept fixed [1, context] full-prefix inputs and emit causal logits.
use crate::native::{NativeBackend, NativeError, NativeResult};
use libloading::Library;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    ffi::{c_char, c_int, c_void, CString},
    marker::PhantomData,
    path::{Path, PathBuf},
    rc::Rc,
};

type Ptr = *mut c_void;
type CreateDelegate = unsafe extern "C" fn(
    *const *const c_char,
    *const *const c_char,
    c_int,
    Option<unsafe extern "C" fn(*const c_char)>,
) -> Ptr;
type DestroyDelegate = unsafe extern "C" fn(Ptr);

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TfliteNpuConfig {
    pub runtime_library: PathBuf,
    pub model: PathBuf,
    pub delegate_library: Option<PathBuf>,
    pub delegate_options: BTreeMap<String, String>,
    pub allow_cpu_fallback: bool,
    pub num_threads: usize,
    pub token_input: usize,
    pub attention_mask_input: Option<usize>,
    pub position_ids_input: Option<usize>,
    pub logits_output: usize,
    pub pad_token_id: u32,
    pub bos_token_id: Option<u32>,
}
impl Default for TfliteNpuConfig {
    fn default() -> Self {
        Self {
            runtime_library: "libtensorflowlite_c.so".into(),
            model: PathBuf::new(),
            delegate_library: None,
            delegate_options: BTreeMap::new(),
            allow_cpu_fallback: false,
            num_threads: 1,
            token_input: 0,
            attention_mask_input: None,
            position_ids_input: None,
            logits_output: 0,
            pad_token_id: 0,
            bos_token_id: None,
        }
    }
}
#[derive(Debug, Clone, Serialize)]
pub struct ExecutionReport {
    pub delegate_active: bool,
    pub fallback_reason: Option<String>,
    pub context_length: usize,
    pub vocabulary_size: usize,
    /// External delegates may partition the graph; activation does not prove NPU coverage.
    pub full_prefix: bool,
}
struct Api {
    _library: Library,
    model_create: unsafe extern "C" fn(*const c_char) -> Ptr,
    model_delete: unsafe extern "C" fn(Ptr),
    options_create: unsafe extern "C" fn() -> Ptr,
    options_delete: unsafe extern "C" fn(Ptr),
    threads: unsafe extern "C" fn(Ptr, c_int),
    delegate: unsafe extern "C" fn(Ptr, Ptr),
    interpreter_create: unsafe extern "C" fn(Ptr, Ptr) -> Ptr,
    interpreter_delete: unsafe extern "C" fn(Ptr),
    allocate: unsafe extern "C" fn(Ptr) -> c_int,
    invoke: unsafe extern "C" fn(Ptr) -> c_int,
    input_count: unsafe extern "C" fn(Ptr) -> c_int,
    output_count: unsafe extern "C" fn(Ptr) -> c_int,
    input: unsafe extern "C" fn(Ptr, c_int) -> Ptr,
    output: unsafe extern "C" fn(Ptr, c_int) -> *const c_void,
    tensor_type: unsafe extern "C" fn(*const c_void) -> c_int,
    dims: unsafe extern "C" fn(*const c_void) -> c_int,
    dim: unsafe extern "C" fn(*const c_void, c_int) -> c_int,
    bytes: unsafe extern "C" fn(*const c_void) -> usize,
    copy_in: unsafe extern "C" fn(Ptr, *const c_void, usize) -> c_int,
    copy_out: unsafe extern "C" fn(*const c_void, Ptr, usize) -> c_int,
}
impl Api {
    unsafe fn load(path: &Path) -> NativeResult<Self> {
        let library = unsafe { Library::new(path) }.map_err(|e| {
            err(format!(
                "cannot load TFLite runtime {}: {e}",
                path.display()
            ))
        })?;
        macro_rules! symbol {
            ($name:literal) => {
                *unsafe { library.get(concat!($name, "\0").as_bytes()) }
                    .map_err(|e| err(format!("missing {}: {e}", $name)))?
            };
        }
        Ok(Self {
            model_create: symbol!("TfLiteModelCreateFromFile"),
            model_delete: symbol!("TfLiteModelDelete"),
            options_create: symbol!("TfLiteInterpreterOptionsCreate"),
            options_delete: symbol!("TfLiteInterpreterOptionsDelete"),
            threads: symbol!("TfLiteInterpreterOptionsSetNumThreads"),
            delegate: symbol!("TfLiteInterpreterOptionsAddDelegate"),
            interpreter_create: symbol!("TfLiteInterpreterCreate"),
            interpreter_delete: symbol!("TfLiteInterpreterDelete"),
            allocate: symbol!("TfLiteInterpreterAllocateTensors"),
            invoke: symbol!("TfLiteInterpreterInvoke"),
            input_count: symbol!("TfLiteInterpreterGetInputTensorCount"),
            output_count: symbol!("TfLiteInterpreterGetOutputTensorCount"),
            input: symbol!("TfLiteInterpreterGetInputTensor"),
            output: symbol!("TfLiteInterpreterGetOutputTensor"),
            tensor_type: symbol!("TfLiteTensorType"),
            dims: symbol!("TfLiteTensorNumDims"),
            dim: symbol!("TfLiteTensorDim"),
            bytes: symbol!("TfLiteTensorByteSize"),
            copy_in: symbol!("TfLiteTensorCopyFromBuffer"),
            copy_out: symbol!("TfLiteTensorCopyToBuffer"),
            _library: library,
        })
    }
}
struct Delegate {
    handle: Ptr,
    destroy: DestroyDelegate,
    _library: Library,
    // Keep plugin option strings alive even for runtimes that retain pointers.
    _keys: Vec<CString>,
    _values: Vec<CString>,
    _key_pointers: Vec<*const c_char>,
    _value_pointers: Vec<*const c_char>,
}
impl Drop for Delegate {
    fn drop(&mut self) {
        unsafe { (self.destroy)(self.handle) }
    }
}
impl Delegate {
    unsafe fn load(config: &TfliteNpuConfig) -> NativeResult<Self> {
        let path = config.delegate_library.as_ref().ok_or_else(|| {
            err("a delegate_library is required unless CPU fallback is explicitly enabled")
        })?;
        let library = unsafe { Library::new(path) }
            .map_err(|e| err(format!("cannot load delegate {}: {e}", path.display())))?;
        let create: CreateDelegate = *unsafe { library.get(b"tflite_plugin_create_delegate\0") }
            .map_err(|e| err(e.to_string()))?;
        let destroy: DestroyDelegate = *unsafe { library.get(b"tflite_plugin_destroy_delegate\0") }
            .map_err(|e| err(e.to_string()))?;
        let keys: Vec<_> = config
            .delegate_options
            .keys()
            .map(|s| cstring(s))
            .collect::<NativeResult<_>>()?;
        let values: Vec<_> = config
            .delegate_options
            .values()
            .map(|s| cstring(s))
            .collect::<NativeResult<_>>()?;
        let kp: Vec<_> = keys.iter().map(|s| s.as_ptr()).collect();
        let vp: Vec<_> = values.iter().map(|s| s.as_ptr()).collect();
        let count = c_int::try_from(kp.len()).map_err(|_| err("too many delegate options"))?;
        let handle = unsafe { create(kp.as_ptr(), vp.as_ptr(), count, None) };
        if handle.is_null() {
            return Err(err("external delegate creation failed"));
        }
        Ok(Self {
            handle,
            destroy,
            _library: library,
            _keys: keys,
            _values: values,
            _key_pointers: kp,
            _value_pointers: vp,
        })
    }
}
/// Must be constructed, invoked and destroyed on one thread. No unsafe Send/Sync.
pub struct TfliteNpuModel {
    api: Api,
    model: Ptr,
    interpreter: Ptr,
    delegate: Option<Delegate>,
    config: TfliteNpuConfig,
    report: ExecutionReport,
    output_rows: usize,
    _thread_affinity: PhantomData<Rc<()>>,
}
impl Drop for TfliteNpuModel {
    fn drop(&mut self) {
        unsafe {
            if !self.interpreter.is_null() {
                (self.api.interpreter_delete)(self.interpreter);
            }
            self.delegate.take();
            if !self.model.is_null() {
                (self.api.model_delete)(self.model);
            }
        }
    }
}
fn err(message: impl Into<String>) -> NativeError {
    NativeError(message.into())
}
fn cstring(s: &str) -> NativeResult<CString> {
    CString::new(s).map_err(|_| err("path/option contains a NUL byte"))
}
fn status(code: c_int, operation: &str) -> NativeResult<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(err(format!("TFLite {operation} failed (status {code})")))
    }
}
impl TfliteNpuModel {
    /// Loading native libraries executes trusted vendor code. Use matching runtime/plugin ABIs.
    pub fn load(config: TfliteNpuConfig) -> NativeResult<Self> {
        if config.num_threads == 0 || config.num_threads > c_int::MAX as usize {
            return Err(err("num_threads must be in 1..=i32::MAX"));
        }
        let path = cstring(
            config
                .model
                .to_str()
                .ok_or_else(|| err("model path must be UTF-8"))?,
        )?;
        let api = unsafe { Api::load(&config.runtime_library) }?;
        let model = unsafe { (api.model_create)(path.as_ptr()) };
        if model.is_null() {
            return Err(err("TFLite could not load the compiled model"));
        }
        let mut result = Self {
            api,
            model,
            interpreter: std::ptr::null_mut(),
            delegate: None,
            config,
            report: ExecutionReport {
                delegate_active: false,
                fallback_reason: None,
                context_length: 0,
                vocabulary_size: 0,
                full_prefix: true,
            },
            output_rows: 0,
            _thread_affinity: PhantomData,
        };
        match unsafe { Delegate::load(&result.config) } {
            Ok(delegate) => result.delegate = Some(delegate),
            Err(error) if result.config.allow_cpu_fallback => {
                result.report.fallback_reason = Some(error.0)
            }
            Err(error) => return Err(error),
        }
        if let Err(error) = result.create_interpreter() {
            if !result.config.allow_cpu_fallback || result.delegate.is_none() {
                return Err(error);
            }
            if !result.interpreter.is_null() {
                unsafe { (result.api.interpreter_delete)(result.interpreter) };
            }
            result.interpreter = std::ptr::null_mut();
            result.delegate.take();
            result.report.fallback_reason = Some(error.0);
            result.create_interpreter()?;
        }
        result.report.delegate_active = result.delegate.is_some();
        result.validate_contract()?;
        Ok(result)
    }
    fn create_interpreter(&mut self) -> NativeResult<()> {
        unsafe {
            let options = (self.api.options_create)();
            if options.is_null() {
                return Err(err("TFLite options allocation failed"));
            }
            (self.api.threads)(options, self.config.num_threads as c_int);
            if let Some(delegate) = &self.delegate {
                (self.api.delegate)(options, delegate.handle);
            }
            self.interpreter = (self.api.interpreter_create)(self.model, options);
            (self.api.options_delete)(options);
            if self.interpreter.is_null() {
                return Err(err("TFLite interpreter creation failed"));
            }
            status((self.api.allocate)(self.interpreter), "tensor allocation")
        }
    }
    fn input(&self, index: usize) -> NativeResult<Ptr> {
        let index = c_int::try_from(index).map_err(|_| err("input index overflow"))?;
        let tensor = unsafe { (self.api.input)(self.interpreter, index) };
        if tensor.is_null() {
            Err(err("missing input tensor"))
        } else {
            Ok(tensor)
        }
    }
    fn shape(&self, tensor: *const c_void) -> NativeResult<Vec<usize>> {
        let rank = unsafe { (self.api.dims)(tensor) };
        if !(1..=3).contains(&rank) {
            return Err(err("expected tensor rank 1..=3"));
        }
        (0..rank)
            .map(|i| {
                let d = unsafe { (self.api.dim)(tensor, i) };
                if d <= 0 {
                    Err(err("only positive fixed tensor dimensions are supported"))
                } else {
                    Ok(d as usize)
                }
            })
            .collect()
    }
    fn output(&self) -> NativeResult<*const c_void> {
        let t = unsafe { (self.api.output)(self.interpreter, self.config.logits_output as c_int) };
        if t.is_null() {
            Err(err("missing logits tensor"))
        } else {
            Ok(t)
        }
    }
    fn validate_contract(&mut self) -> NativeResult<()> {
        let count = unsafe { (self.api.input_count)(self.interpreter) };
        let mut indices = vec![self.config.token_input];
        indices.extend(self.config.attention_mask_input);
        indices.extend(self.config.position_ids_input);
        indices.sort_unstable();
        if count <= 0 || indices != (0..count as usize).collect::<Vec<_>>() {
            return Err(err("input mappings must cover every input exactly once"));
        }
        let shape = self.shape(self.input(self.config.token_input)?)?;
        if shape.len() != 2 || shape[0] != 1 {
            return Err(err("token input must have fixed shape [1, context]"));
        }
        self.report.context_length = shape[1];
        for index in indices {
            let t = self.input(index)?;
            if self.shape(t)? != shape {
                return Err(err("all inputs must have shape [1, context]"));
            }
            let size = match unsafe { (self.api.tensor_type)(t) } {
                2 => 4,
                4 => 8,
                _ => {
                    return Err(err(
                        "token, mask and position inputs must be int32 or int64",
                    ))
                }
            };
            if shape[1].checked_mul(size) != Some(unsafe { (self.api.bytes)(t) }) {
                return Err(err("input tensor byte size does not match shape"));
            }
        }
        let count = unsafe { (self.api.output_count)(self.interpreter) };
        if count <= 0 || self.config.logits_output >= count as usize {
            return Err(err("logits output index is out of range"));
        }
        let t = self.output()?;
        if unsafe { (self.api.tensor_type)(t) } != 1 {
            return Err(err(
                "logits output must be float32; export a dequantized output",
            ));
        }
        let shape = self.shape(t)?;
        let (rows,vocab)=match shape.as_slice(){[1,v] =>(1,*v),[s,v] if *s==self.report.context_length=>(*s,*v),[1,s,v] if *s==self.report.context_length=>(*s,*v),_=>return Err(err("logits must have shape [1, vocabulary], [context, vocabulary] or [1, context, vocabulary]"))};
        if rows.checked_mul(vocab).and_then(|n| n.checked_mul(4))
            != Some(unsafe { (self.api.bytes)(t) })
        {
            return Err(err("logits byte size does not match shape"));
        }
        self.output_rows = rows;
        self.report.vocabulary_size = vocab;
        if self.config.pad_token_id as usize >= vocab
            || self
                .config
                .bos_token_id
                .is_some_and(|id| id as usize >= vocab)
        {
            return Err(err("pad/BOS token ID exceeds the model vocabulary"));
        }
        Ok(())
    }
    pub fn report(&self) -> &ExecutionReport {
        &self.report
    }
    fn write_input(&self, index: usize, values: &[i64]) -> NativeResult<()> {
        let t = self.input(index)?;
        if unsafe { (self.api.tensor_type)(t) } == 2 {
            let values: Vec<i32> = values
                .iter()
                .map(|v| i32::try_from(*v).map_err(|_| err("input value exceeds int32")))
                .collect::<NativeResult<_>>()?;
            status(
                unsafe { (self.api.copy_in)(t, values.as_ptr().cast(), values.len() * 4) },
                "input copy",
            )
        } else {
            status(
                unsafe {
                    (self.api.copy_in)(t, values.as_ptr().cast(), std::mem::size_of_val(values))
                },
                "input copy",
            )
        }
    }
    /// Evaluate the complete prefix. Padded positions are masked; no persistent KV cache.
    pub fn forward(&mut self, tokens: &[u32]) -> NativeResult<Vec<f32>> {
        let context = self.report.context_length;
        if tokens
            .iter()
            .any(|id| *id as usize >= self.report.vocabulary_size)
        {
            return Err(err("token ID exceeds the model vocabulary"));
        }
        if tokens.is_empty() || tokens.len() > context {
            return Err(err("prefix must contain 1..=context_length tokens"));
        }
        let mut ids = vec![self.config.pad_token_id as i64; context];
        for (out, id) in ids.iter_mut().zip(tokens) {
            *out = *id as i64;
        }
        self.write_input(self.config.token_input, &ids)?;
        if let Some(index) = self.config.attention_mask_input {
            self.write_input(
                index,
                &(0..context)
                    .map(|i| i64::from(i < tokens.len()))
                    .collect::<Vec<_>>(),
            )?;
        }
        if let Some(index) = self.config.position_ids_input {
            self.write_input(index, &(0..context).map(|i| i as i64).collect::<Vec<_>>())?;
        }
        status(unsafe { (self.api.invoke)(self.interpreter) }, "invoke")?;
        let mut output = vec![0.0f32; self.output_rows * self.report.vocabulary_size];
        status(
            unsafe {
                (self.api.copy_out)(self.output()?, output.as_mut_ptr().cast(), output.len() * 4)
            },
            "output copy",
        )?;
        let row = if self.output_rows == 1 {
            0
        } else {
            tokens.len() - 1
        };
        let start = row * self.report.vocabulary_size;
        let logits = output[start..start + self.report.vocabulary_size].to_vec();
        if logits.iter().any(|v| !v.is_finite()) {
            return Err(err("model emitted non-finite logits"));
        }
        Ok(logits)
    }
}
pub struct TfliteNpuBackend {
    model: TfliteNpuModel,
    tokenizer: tokenizers::Tokenizer,
}
impl TfliteNpuBackend {
    pub fn from_files(config: TfliteNpuConfig, tokenizer: impl AsRef<Path>) -> NativeResult<Self> {
        let tokenizer = crate::native::load_tokenizer(tokenizer.as_ref())?;
        let model = TfliteNpuModel::load(config)?;
        if tokenizer
            .get_vocab(true)
            .values()
            .any(|id| *id as usize >= model.report.vocabulary_size)
        {
            return Err(err("tokenizer IDs exceed the model vocabulary"));
        }
        Ok(Self { model, tokenizer })
    }
    pub fn report(&self) -> &ExecutionReport {
        self.model.report()
    }
}
impl NativeBackend for TfliteNpuBackend {
    fn encode(&self, text: &str) -> NativeResult<Vec<u32>> {
        let already_bos = self
            .model
            .config
            .bos_token_id
            .and_then(|id| self.tokenizer.id_to_token(id))
            .is_some_and(|bos| text.trim_start().starts_with(&bos));
        self.tokenizer
            .encode(text, !already_bos)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| err(e.to_string()))
    }
    fn decode(&self, tokens: &[u32]) -> NativeResult<String> {
        self.tokenizer
            .decode(tokens, true)
            .map_err(|e| err(e.to_string()))
    }
    fn logits(
        &mut self,
        _sequence_id: &str,
        tokens: &[u32],
        _position: usize,
    ) -> NativeResult<Vec<f32>> {
        self.model.forward(tokens)
    }
}

impl crate::native::NativeModel for TfliteNpuModel {
    fn forward(
        &mut self,
        _sequence_id: &str,
        tokens: &[u32],
        _position: usize,
    ) -> NativeResult<Vec<f32>> {
        TfliteNpuModel::forward(self, tokens)
    }
}
