//! Narrow demonstrations of experimental Candle APIs and upstream alternatives.
//! See docs/handbook.md for limits; no downloads or persistent output files.
#[cfg(feature = "candle")]
fn main() -> anyhow::Result<()> {
    use candlelighter::densetypes::DenseType;
    use candlelighter::layer::sparsemoe::{SparseMoE, SparseMoETrait};
    use candlelighter::liquid::*;
    use candlelighter::prelude::*;
    let dev = Device::Cpu;
    let vars = VarMap::new();
    let kernel = Tensor::ones((1, 1, 2, 2), DType::F32, &dev)?;
    let conv = Conv::new(kernel, 2, 0, 1, 1, 1, &dev, &vars, "conv2d".into());
    assert_eq!(
        conv.forward(Tensor::ones((1, 1, 4, 4), DType::F32, &dev)?)
            .dims(),
        &[1, 1, 3, 3]
    );
    let x = Tensor::new(&[[1f32, 2., 3.]], &dev)?;
    let mask = Tensor::new(&[[1f32, 0., 1.]], &dev)?;
    assert_eq!(x.mul(&mask)?.to_vec2::<f32>()?, vec![vec![1., 0., 3.]]);
    let dropout = candle_nn::Dropout::new(0.1);
    assert_eq!(
        dropout.forward_t(&x, false)?.to_vec2::<f32>()?,
        x.to_vec2::<f32>()?
    );
    let _training = dropout.forward_t(&x, true)?;
    let config = candle_nn::ParamsAdamW {
        lr: 1e-3,
        weight_decay: 0.01,
        ..Default::default()
    };
    let _optimizer = candle_nn::AdamW::new(vars.all_vars(), config)?;
    let logits = Tensor::new(&[[2f32, -1.], [-1., 2.]], &dev)?;
    let labels = Tensor::new(&[0u32, 1], &dev)?;
    assert!(candle_nn::loss::cross_entropy(&logits, &labels)?.to_scalar::<f32>()? > 0.);

    let attention = SelfAttention::new(1, 1, 1, 1, &dev, &VarMap::new(), "attention".into());
    assert_eq!(
        attention
            .forward(Tensor::ones((1, 1), DType::F32, &dev)?)
            .dims(),
        &[1, 1]
    );
    let moe = SparseMoE::new(2, 4, 4, &dev, &VarMap::new(), "moe".into());
    assert_eq!(
        moe.forward(Tensor::ones((1, 4), DType::F32, &dev)?).dims(),
        &[1, 4]
    );
    // Only construction: these variants are not a working adapter trainer.
    for kind in [DenseType::LORA, DenseType::DORA] {
        let rank = Tensor::zeros((2, 2), DType::F32, &dev)?;
        let _adapter = Dense::new2(
            2,
            2,
            Activations::Linear,
            kind,
            rank,
            1.,
            &dev,
            &VarMap::new(),
            "adapter".into(),
        );
    }
    let branches: Vec<Box<dyn Trainable>> = vec![
        Box::new(Dense::new(
            2,
            2,
            Activations::Linear,
            &dev,
            &vars,
            "branch_a".into(),
        )),
        Box::new(Dense::new(
            2,
            2,
            Activations::Linear,
            &dev,
            &vars,
            "branch_b".into(),
        )),
    ];
    let _split = ParallelModel::new(ParallelModelType::Split, &dev, vars, branches);
    // Explicit averaging is a working ensemble alternative.
    let a = Tensor::new(&[[1f32, 3.]], &dev)?;
    let b = Tensor::new(&[[3f32, 5.]], &dev)?;
    assert_eq!(((a + b)? * 0.5)?.to_vec2::<f32>()?, vec![vec![2., 4.]]);

    for solver in [OdeSolver::Euler, OdeSolver::Heun, OdeSolver::RungeKutta4] {
        let ode = NeuralOde::new(|_, state: &[f32]| vec![-state[0]], solver);
        let state = ode.solve(&[1.], 0., 1., 20)?;
        assert!((state[0] - (-1f32).exp()).abs() < 0.02);
    }
    let cell = LiquidCell::new(2, 4)?.with_solver(OdeSolver::Heun);
    assert_eq!(cell.step(&[1., 0.], &[0.; 4], 0.1)?.len(), 4);
    println!("experimental handbook recipes completed (see documented limits)");
    Ok(())
}

#[cfg(not(feature = "candle"))]
fn main() {
    eprintln!("run this example with the candle feature enabled");
}
