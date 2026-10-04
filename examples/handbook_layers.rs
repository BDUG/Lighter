//! Small CPU examples for the usage handbook; no downloads or output files.
#[cfg(feature = "candle")]
fn main() -> anyhow::Result<()> {
    use candlelighter::embeddingtypes::EmbeddingType;
    use candlelighter::layer::embeddinglayer::{Embed, EmbeddingLayerTrait};
    use candlelighter::prelude::*;
    use candlelighter::preprocessing::features::{Features, FeaturesTrait};
    use candlelighter::recurrenttypes::RecurrentType;

    let dev = Device::Cpu;
    let mut features = Features::new(dev.clone());
    features.add_feature_1_d(vec![1., 2.]);
    features.add_feature_1_d(vec![3., 4.]);
    assert_eq!(features.get_data_tensor().dims(), &[2, 1, 2]);

    let values = Tensor::new(&[[1f32, 2., 3.]], &dev)?;
    let scaling = FeatureScaling::new(values);
    assert_eq!(
        scaling.min_max_normalization().to_vec2::<f32>()?,
        vec![vec![0., 0.5, 1.]]
    );
    println!("z-score: {:?}", scaling.z_score().to_vec2::<f32>()?);
    let restored = scaling.min_max_normalization_reverse(scaling.min_max_normalization());
    assert_eq!(restored.to_vec1::<f32>()?, vec![1., 2., 3.]);

    let vars = VarMap::new();
    let x = Tensor::new(&[[1f32, 2.]], &dev)?;
    for (i, activation) in [
        Activations::Linear,
        Activations::Relu,
        Activations::Silu,
        Activations::Sigmoid,
        Activations::Softmax,
    ]
    .into_iter()
    .enumerate()
    {
        let dense = Dense::new(3, 2, activation, &dev, &vars, format!("dense_{i}"));
        assert_eq!(dense.forward(x.clone()).dims(), &[1, 3]);
    }

    let kernel = Tensor::ones((1, 1, 2), DType::F32, &dev)?;
    let conv = Conv::new(kernel, 1, 0, 1, 1, 1, &dev, &vars, "conv".into());
    let signal = Tensor::new(&[[[1f32, 2., 3., 4.]]], &dev)?;
    assert_eq!(
        conv.forward(signal).flatten_all()?.to_vec1::<f32>()?,
        vec![3., 5., 7.]
    );

    let image = Tensor::new(&[[[1f32, 2.], [3., 4.]]], &dev)?;
    // Current wrapper behavior: MAX averages; AVERAGE takes the maximum.
    for (kind, expected) in [(PoolingType::MAX, 2.5f32), (PoolingType::AVERAGE, 4.)] {
        let pool = Pooling::new(kind, 2, 2, &dev, &vars, "pool".into());
        assert_eq!(
            pool.forward(image.clone())
                .flatten_all()?
                .to_vec1::<f32>()?,
            vec![expected]
        );
    }
    let flat = Flatten::new(&dev, &vars, "flatten".into());
    assert_eq!(flat.forward(image).dims(), &[1, 4]);
    // Normalization's current forward method discards the normalized tensor.
    let normalization = Normalization::new(1, &dev, &vars, "norm".into());
    assert_eq!(
        normalization.forward(x.clone()).to_vec2::<f32>()?,
        x.to_vec2::<f32>()?
    );
    let normalized = x.broadcast_div(&x.sqr()?.sum_keepdim(1)?.sqrt()?)?;
    println!(
        "explicit L2 normalization: {:?}",
        normalized.to_vec2::<f32>()?
    );

    for kind in [RecurrentType::LSTM, RecurrentType::GRU] {
        // Separate maps: the wrapper does not namespace its recurrent weights.
        let rnn = Recurrent::new(kind, 2, 3, &dev, &VarMap::new(), "rnn".into());
        assert_eq!(rnn.forward(x.clone()).dims(), &[1, 3]);
    }
    let embedding = Embed::new(
        EmbeddingType::Standard,
        8,
        3,
        &dev,
        &VarMap::new(),
        "embed".into(),
    );
    assert_eq!(
        embedding.forward(Tensor::new(&[0u32, 2, 7], &dev)?).dims(),
        &[3, 3]
    );

    // An autoencoder architecture assembled from existing dense layers.
    let ae_vars = VarMap::new();
    let autoencoder = SequentialModel::new(
        ae_vars.clone(),
        vec![
            Box::new(Dense::new(
                1,
                2,
                Activations::Relu,
                &dev,
                &ae_vars,
                "encoder".into(),
            )),
            Box::new(Dense::new(
                2,
                1,
                Activations::Linear,
                &dev,
                &ae_vars,
                "decoder".into(),
            )),
        ],
    );
    assert_eq!(autoencoder.forward(x).dims(), &[1, 2]);

    // Small supervised regression using the existing fit/predict API.
    let train_vars = VarMap::new();
    let mut model = SequentialModel::new(
        train_vars.clone(),
        vec![Box::new(Dense::new(
            1,
            2,
            Activations::Linear,
            &dev,
            &train_vars,
            "regression".into(),
        ))],
    );
    model.compile(Optimizers::SGD(0.01), Loss::MSE);
    let inputs = Tensor::new(&[[[1f32, 2.]], [[2., 3.]]], &dev)?;
    let targets = Tensor::new(&[[[3f32]], [[5.]]], &dev)?;
    model.fit(inputs.clone(), targets, 2, false);
    assert_eq!(model.predict(&inputs).unwrap().len(), 2);
    println!("handbook layer examples completed");
    Ok(())
}

#[cfg(not(feature = "candle"))]
fn main() {
    eprintln!("run this example with the candle feature enabled");
}
