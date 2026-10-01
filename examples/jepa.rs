//! Minimal I-JEPA and V-JEPA forward passes.
use candlelighter::jepa::{
    BlockMask, IJepa, JepaConfig, JepaGeometry, JepaTrainer, MeanEncoder, VJepa,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let target = [BlockMask {
        row: 0,
        col: 1,
        height: 1,
        width: 1,
    }];
    let pixels: Vec<f32> = (0..16).map(|x| x as f32 / 15.0).collect();

    let image_model = IJepa::new(MeanEncoder::new(4)?, 2, 2)?;
    let image = image_model.forward(&pixels, 4, 4, 1, &target)?;
    println!(
        "I-JEPA targets: {}, MSE: {:.6}",
        image.targets.len(),
        image.loss
    );

    let mut video_pixels = pixels.clone();
    video_pixels.extend(pixels.iter().rev());
    let video_model = VJepa::new(MeanEncoder::new(4)?, 2, 2)?;
    let video = video_model.forward(&video_pixels, 2, 4, 4, 1, &target)?;
    println!(
        "V-JEPA tube targets: {}, MSE: {:.6}",
        video.targets.len(),
        video.loss
    );

    // A real training step uses an online encoder and predictor, stop-gradient
    // target representations, and an EMA update of the target encoder.
    let mut trainer = JepaTrainer::new(
        JepaConfig {
            patch_elements: 4,
            latent_dim: 8,
            learning_rate: 1e-2,
            target_momentum: 0.99,
        },
        42,
    )?;
    let geometry = JepaGeometry {
        height: 4,
        width: 4,
        channels: 1,
        patch_height: 2,
        patch_width: 2,
    };
    for epoch in 0..5 {
        let loss = trainer.train_image(&pixels, geometry, &target)?;
        println!("training step {epoch}: {loss:.6}");
    }
    Ok(())
}
