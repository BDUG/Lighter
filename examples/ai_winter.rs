//! Train XOR, save weights, then run inference in a separate process.
use candle_core::{Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarMap};
use candlelighter::prelude::{Activations, Dense, DenseLayerTrait, Trainable};
use rand::{rngs::StdRng, Rng, SeedableRng};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 || !matches!(args[0].as_str(), "train" | "infer") {
        return Err("usage: ai_winter {train|infer} CHECKPOINT.safetensors".into());
    }
    let device = Device::Cpu;
    let mut vars = VarMap::new();
    let hidden = Dense::new(8, 2, Activations::Sigmoid, &device, &vars, "hidden".into());
    let output = Dense::new(1, 8, Activations::Sigmoid, &device, &vars, "output".into());
    let inputs = Tensor::new(&[[0f32, 0.], [0., 1.], [1., 0.], [1., 1.]], &device)?;
    let targets = Tensor::new(&[[0f32], [1.], [1.], [0.]], &device)?;
    let forward = |x: Tensor| output.forward(hidden.forward(x));

    if args[0] == "train" {
        // Fixed seeded values make this CPU demonstration reproducible.
        let mut rng = StdRng::seed_from_u64(42);
        for (name, shape) in [
            ("hidden.weight", vec![8, 2]),
            ("hidden.bias", vec![8]),
            ("output.weight", vec![1, 8]),
            ("output.bias", vec![1]),
        ] {
            let count: usize = shape.iter().product();
            let values: Vec<f32> = (0..count).map(|_| rng.gen_range(-1.0..1.0)).collect();
            vars.set_one(name, Tensor::from_vec(values, shape, &device)?)?;
        }
        // Keep optimizer state across steps; train all four rows in one batch.
        let mut optimizer = AdamW::new(
            vars.all_vars(),
            ParamsAdamW {
                lr: 0.05,
                weight_decay: 0.0,
                ..Default::default()
            },
        )?;
        let initial =
            candle_nn::loss::mse(&forward(inputs.clone()), &targets)?.to_scalar::<f32>()?;
        println!("initial MSE: {initial:.6}");
        let mut converged = false;
        for step in 1..=5000 {
            let loss = candle_nn::loss::mse(&forward(inputs.clone()), &targets)?;
            optimizer.backward_step(&loss)?;
            if step % 100 == 0 {
                let value =
                    candle_nn::loss::mse(&forward(inputs.clone()), &targets)?.to_scalar::<f32>()?;
                println!("step {step:4}: MSE={value:.6}");
                if value < 0.001 {
                    converged = true;
                    break;
                }
            }
        }
        if !converged {
            return Err("XOR training did not reach the target loss".into());
        }
        vars.save(&args[1])?;
        println!("saved {}", args[1]);
    } else {
        // Reconstruct the same variable names/shapes before loading saved weights.
        vars.load(&args[1])?;
        println!("loaded {}", args[1]);
    }

    let probabilities = forward(inputs).to_vec2::<f32>()?;
    println!("a b | probability | XOR");
    for (index, expected) in [0u8, 1, 1, 0].into_iter().enumerate() {
        let probability = probabilities[index][0];
        let prediction = u8::from(probability >= 0.5);
        println!(
            "{} {} | {:.4}      | {}",
            index / 2,
            index % 2,
            probability,
            prediction
        );
        if !probability.is_finite() || prediction != expected {
            return Err(format!("incorrect XOR prediction for row {index}").into());
        }
    }
    Ok(())
}
