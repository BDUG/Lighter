//! Complete Rust training/persistence counterparts to the Python Keras examples.
#[cfg(feature = "candle")]
fn main() -> anyhow::Result<()> {
    use candlelighter::prelude::*;
    let dev = Device::Cpu;
    let vars = VarMap::new();
    let mut model = SequentialModel::new(
        vars.clone(),
        vec![
            Box::new(Dense::new(
                4,
                2,
                Activations::Relu,
                &dev,
                &vars,
                "hidden".into(),
            )),
            Box::new(Dense::new(
                1,
                4,
                Activations::Linear,
                &dev,
                &vars,
                "output".into(),
            )),
        ],
    );
    let x = Tensor::new(&[[[1f32, 2.]], [[2., 3.]], [[3., 4.]], [[4., 5.]]], &dev)?;
    let y = Tensor::new(&[[[3f32]], [[5.]], [[7.]], [[9.]]], &dev)?;
    model.compile(Optimizers::SGD(0.01), Loss::MSE);
    model.fit(x.clone(), y, 3, false);
    let predictions = model.predict(&x).unwrap();
    assert_eq!(predictions.len(), 4);
    for prediction in predictions {
        assert_eq!(prediction.dims(), &[1, 1]);
    }

    // A custom classification loop supplies a loss the Lighter fit wrapper rejects.
    let mut class_vars = VarMap::new();
    let classifier = Dense::new(
        2,
        2,
        Activations::Linear,
        &dev,
        &class_vars,
        "classifier".into(),
    );
    let inputs = Tensor::new(&[[1f32, 2.], [2., 3.], [3., 4.], [4., 5.]], &dev)?;
    let labels = Tensor::new(&[0u32, 0, 1, 1], &dev)?;
    let mut optimizer = candle_nn::AdamW::new(
        class_vars.all_vars(),
        candle_nn::ParamsAdamW {
            lr: 1e-3,
            weight_decay: 0.01,
            ..Default::default()
        },
    )?;
    for _ in 0..3 {
        let logits = classifier.forward(inputs.clone());
        let loss = candle_nn::loss::cross_entropy(&logits, &labels)?;
        assert!(loss.to_scalar::<f32>()?.is_finite());
        optimizer.backward_step(&loss)?;
    }
    // Restore into the SAME registered variables, preserving tensor shapes.
    let before = classifier.forward(inputs.clone()).to_vec2::<f32>()?;
    let directory =
        std::env::temp_dir().join(format!("lighter-handbook-keras-{}", std::process::id()));
    std::fs::create_dir(&directory)?; // Refuse to overwrite an existing directory.
    let path = directory.join("classification.safetensors");
    let result = (|| -> anyhow::Result<()> {
        class_vars.save(&path)?;
        // Destroy the in-memory weights so equality demonstrates actual restoration.
        for variable in class_vars.all_vars() {
            variable.set(&variable.as_tensor().zeros_like()?)?;
        }
        assert_ne!(before, classifier.forward(inputs.clone()).to_vec2::<f32>()?);
        class_vars.load(&path)?;
        assert_eq!(before, classifier.forward(inputs).to_vec2::<f32>()?);
        Ok(())
    })();
    if path.exists() {
        std::fs::remove_file(&path)?;
    }
    std::fs::remove_dir(&directory)?;
    result?;
    println!(
        "Rust regression, custom classification and prediction-equivalent weight reload completed"
    );
    Ok(())
}
#[cfg(not(feature = "candle"))]
fn main() {
    eprintln!("enable the candle feature");
}
