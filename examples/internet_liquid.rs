//! Liquid-network forecasting on a public internet time-series dataset.
//!
//! The daily minimum-temperature dataset is the same real-world forecasting
//! dataset published by Jason Brownlee at
//! https://github.com/jbrownlee/Datasets. The continuous-time model follows the
//! official LTC examples at https://github.com/raminmh/liquid_time_constant_networks.

#[cfg(feature = "native")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use candlelighter::liquid::{CfcCell, Lfm, NeuralOde, OdeSolver};

    const DATA_URL: &str =
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv";
    let csv = ureq::get(DATA_URL).call()?.into_string()?;
    let temperatures = csv
        .lines()
        .skip(1)
        .filter_map(|line| {
            line.rsplit_once(',')?
                .1
                .trim_matches('"')
                .parse::<f32>()
                .ok()
        })
        .take(365)
        .collect::<Vec<_>>();
    if temperatures.len() < 2 {
        return Err("the downloaded dataset did not contain enough samples".into());
    }
    let mean = temperatures.iter().sum::<f32>() / temperatures.len() as f32;
    let scale = temperatures
        .iter()
        .map(|value| (value - mean).abs())
        .fold(0.0f32, f32::max)
        .max(f32::EPSILON);
    let normalized = temperatures
        .iter()
        .map(|x| (x - mean) / scale)
        .collect::<Vec<_>>();
    let inputs = normalized[..normalized.len() - 1]
        .iter()
        .map(|x| vec![*x])
        .collect::<Vec<_>>();
    let targets = normalized[1..].iter().map(|x| vec![*x]).collect::<Vec<_>>();

    let mut lfm = Lfm::new(1, &[12, 6], 1)?;
    for epoch in 0..25 {
        let loss = lfm.fit_readout(&inputs, &targets, 1.0, 0.01)?;
        if epoch % 5 == 0 || epoch == 24 {
            println!("LFM epoch {epoch:2}: normalized MSE={loss:.6}");
        }
    }

    // CfC directly accepts irregular elapsed times (e.g. missing observations).
    let cfc = CfcCell::new(1, 6)?;
    let elapsed = (0..inputs.len())
        .map(|i| if i % 17 == 0 { 2.0 } else { 1.0 })
        .collect::<Vec<_>>();
    let states = cfc.forward_irregular(&inputs, &elapsed)?;
    println!("CfC processed {} irregular observations", states.len());

    // A Neural ODE baseline models exponential relaxation toward the dataset mean.
    let relaxation = NeuralOde::new(
        |_, state: &[f32]| vec![-0.15 * state[0]],
        OdeSolver::RungeKutta4,
    );
    let forecast = relaxation.solve(&[normalized[0]], 0.0, 7.0, 28)?;
    println!(
        "Neural ODE seven-day forecast: {:.2} C",
        forecast[0] * scale + mean
    );
    Ok(())
}

#[cfg(not(feature = "native"))]
fn main() {
    eprintln!("enable this internet-backed example with --no-default-features --features native");
}
