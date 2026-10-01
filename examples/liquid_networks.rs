//! Neural ODE and liquid foundation model examples.
use candlelighter::liquid::{CfcCell, Lfm, NeuralOde, OdeSolver};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let decay = NeuralOde::new(|_, state: &[f32]| vec![-state[0]], OdeSolver::RungeKutta4);
    let state = decay.solve(&[1.0], 0.0, 1.0, 20)?;
    println!(
        "Neural ODE y(1): {:.6} (expected {:.6})",
        state[0],
        (-1.0f32).exp()
    );

    let mut model = Lfm::new(2, &[8, 4], 1)?;
    let samples = vec![vec![1.0, 0.0], vec![0.5, 0.5], vec![0.0, 1.0]];
    println!("LFM outputs: {:?}", model.forward(&samples, 0.1)?);
    let targets = vec![vec![1.0], vec![0.5], vec![0.0]];
    println!(
        "LFM readout training loss: {:.6}",
        model.fit_readout(&samples, &targets, 0.1, 0.05)?
    );

    let cfc = CfcCell::new(2, 4)?;
    let elapsed = [0.1, 0.4, 1.2];
    println!(
        "CfC irregular-time states: {:?}",
        cfc.forward_irregular(&samples, &elapsed)?
    );
    Ok(())
}
