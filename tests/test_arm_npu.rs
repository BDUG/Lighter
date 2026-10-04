#![cfg(all(feature = "arm-npu", target_os = "linux"))]
use candlelighter::arm_npu::{TfliteNpuConfig, TfliteNpuModel};
use std::{path::PathBuf, process::Command};
#[test]
fn external_delegate_abi_inference_fallback_and_cleanup() {
    let dir = std::env::temp_dir().join(format!("lighter-tflite-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let library = dir.join("libshim.so");
    let result = Command::new("cc")
        .args([
            "-shared",
            "-fPIC",
            "-O2",
            "tests/fixtures/tflite_runtime.c",
            "-o",
        ])
        .arg(&library)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let config = TfliteNpuConfig {
        runtime_library: library.clone(),
        model: PathBuf::from("test.tflite"),
        delegate_library: Some(library.clone()),
        attention_mask_input: Some(1),
        position_ids_input: Some(2),
        ..Default::default()
    };
    // Retain one library handle to observe destruction counters after models drop.
    let observer = unsafe { libloading::Library::new(&library) }.unwrap();
    let handles: libloading::Symbol<unsafe extern "C" fn() -> i32> =
        unsafe { observer.get(b"lighter_test_live_handles\0") }.unwrap();
    {
        let mut model = TfliteNpuModel::load(config.clone()).unwrap();
        assert!(model.report().delegate_active);
        assert_eq!(model.report().context_length, 4);
        assert_eq!(model.forward(&[1, 2]).unwrap(), vec![22., 23., 24., 25.]);
        assert_eq!(model.forward(&[3]).unwrap(), vec![31., 32., 33., 34.]);
        assert!(model.forward(&[]).is_err());
        assert!(model.forward(&[0; 5]).is_err());
    }
    assert_eq!(unsafe { handles() }, 0);
    {
        use candlelighter::{
            arm_npu::TfliteNpuBackend,
            native::{GenerateRequest, NativeEngine, SamplingParams},
        };
        let tokenizer = dir.join("tokenizer.json");
        std::fs::write(&tokenizer,serde_json::json!({"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"<unk>":0,"hello":1,"eos":2,"other":3},"unk_token":"<unk>"}}).to_string()).unwrap();
        let backend = TfliteNpuBackend::from_files(config.clone(), &tokenizer).unwrap();
        let mut engine = NativeEngine::new(backend, 2, vec![2]).unwrap();
        for id in ["first", "second"] {
            engine
                .submit(GenerateRequest {
                    id: id.into(),
                    prompt: "hello".into(),
                    sampling: SamplingParams {
                        temperature: 0.,
                        max_tokens: 2,
                        ..Default::default()
                    },
                    constraint: None,
                })
                .unwrap();
        }
        let results = engine.run_to_completion().unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.text == "other other"));
    }
    assert_eq!(unsafe { handles() }, 0);
    let mut failed = config.clone();
    failed.delegate_options.insert("fail".into(), "true".into());
    assert!(TfliteNpuModel::load(failed.clone()).is_err());
    assert_eq!(unsafe { handles() }, 0);
    failed.allow_cpu_fallback = true;
    {
        let model = TfliteNpuModel::load(failed).unwrap();
        assert!(!model.report().delegate_active);
        assert!(model.report().fallback_reason.is_some());
    }
    for path in ["allocate-fail.tflite", "bad-shape.tflite"] {
        let mut c = config.clone();
        c.model = path.into();
        assert!(TfliteNpuModel::load(c).is_err());
        assert_eq!(unsafe { handles() }, 0);
    }
    let mut c = config.clone();
    c.model = "allocate-fail.tflite".into();
    c.allow_cpu_fallback = true;
    {
        let mut model = TfliteNpuModel::load(c).unwrap();
        assert!(!model.report().delegate_active);
        assert!(model.forward(&[1]).is_ok());
    }
    let mut c = config.clone();
    c.model = "invoke-fail.tflite".into();
    {
        let mut model = TfliteNpuModel::load(c).unwrap();
        assert!(model.forward(&[1]).is_err());
    }
    let mut c = config.clone();
    c.position_ids_input = Some(1);
    assert!(TfliteNpuModel::load(c).is_err());
    let mut c = config;
    c.delegate_library = None;
    assert!(TfliteNpuModel::load(c).is_err());
    assert_eq!(unsafe { handles() }, 0);
    drop(handles);
    drop(observer);
    std::fs::remove_dir_all(dir).unwrap();
}
