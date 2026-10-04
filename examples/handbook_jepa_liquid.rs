//! Complete custom encoder, image/video training and stateful continuous-time recipes.
use candlelighter::jepa::*;
use candlelighter::liquid::*;

struct SummaryEncoder;
impl PatchEncoder for SummaryEncoder {
    fn latent_dim(&self) -> usize {
        2
    }
    fn encode(&self, patch: &[f32]) -> Result<Vec<f32>, JepaError> {
        if patch.is_empty() {
            return Err(JepaError("empty patch".into()));
        }
        let mean = patch.iter().sum::<f32>() / patch.len() as f32;
        let energy = patch.iter().map(|x| x * x).sum::<f32>() / patch.len() as f32;
        Ok(vec![mean, energy])
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pixels: Vec<f32> = (0..16).map(|value| value as f32 / 15.).collect();
    let mask = [BlockMask {
        row: 0,
        col: 1,
        height: 1,
        width: 1,
    }];
    let image = IJepa::new(SummaryEncoder, 2, 2)?.forward(&pixels, 4, 4, 1, &mask)?;
    assert!(image.loss.is_finite());
    let mut video = pixels.clone();
    video.extend(pixels.iter().rev().copied());
    let prediction = VJepa::new(SummaryEncoder, 2, 2)?.forward(&video, 2, 4, 4, 1, &mask)?;
    assert!(prediction.loss.is_finite());
    let geometry = JepaGeometry {
        height: 4,
        width: 4,
        channels: 1,
        patch_height: 2,
        patch_width: 2,
    };
    let mut trainer = JepaTrainer::new(
        JepaConfig {
            patch_elements: 4,
            latent_dim: 8,
            learning_rate: 0.01,
            target_momentum: 0.99,
        },
        42,
    )?;
    for _ in 0..3 {
        assert!(trainer.train_image(&pixels, geometry, &mask)?.is_finite());
        assert!(trainer.train_video(&video, 2, geometry, &mask)?.is_finite());
    }
    assert_eq!(trainer.steps(), 6);

    for solver in [OdeSolver::Euler, OdeSolver::Heun, OdeSolver::RungeKutta4] {
        let ode = NeuralOde::new(|_, state: &[f32]| vec![-state[0]], solver);
        assert!((ode.solve(&[1.], 0., 1., 20)?[0] - (-1f32).exp()).abs() < 0.02);
    }
    let cell = LiquidCell::from_parameters(
        1,
        2,
        vec![0.2, -0.1],
        vec![0.; 4],
        vec![0.; 2],
        vec![1., 2.],
        OdeSolver::Heun,
    )?;
    let mut hidden = vec![0.; cell.hidden_size()];
    for (input, elapsed) in [(1., 0.1), (0.5, 0.4), (0., 1.2)] {
        hidden = cell.step(&[input], &hidden, elapsed)?;
        assert!(hidden.iter().all(|x| x.is_finite()));
    }
    let samples = vec![vec![1., 0.], vec![0.5, 0.5], vec![0., 1.]];
    let elapsed = [0.1, 0.4, 1.2];
    let cfc = CfcCell::new(2, 4)?;
    let mut state = vec![0.; 4];
    let mut streamed = vec![];
    for (input, dt) in samples.iter().zip(elapsed) {
        state = cfc.step(input, &state, dt)?;
        streamed.push(state.clone());
    }
    assert_eq!(streamed, cfc.forward_irregular(&samples, &elapsed)?);
    let mut lfm = Lfm::new(2, &[8, 4], 1)?;
    let mut states = lfm.zero_state();
    for input in &samples {
        assert_eq!(lfm.step(input, &mut states, 0.1)?.len(), 1);
    }
    let targets = vec![vec![1.], vec![0.5], vec![0.]];
    for _ in 0..5 {
        assert!(lfm.fit_readout(&samples, &targets, 0.1, 0.05)?.is_finite());
    }
    assert_eq!(lfm.forward(&samples, 0.1)?.len(), samples.len());
    println!("custom JEPA encoder, six image/video updates and stateful liquid examples completed");
    Ok(())
}
