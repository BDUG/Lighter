//! I-JEPA and V-JEPA training on an image downloaded from the internet.
//!
//! The architecture and masking follow the official Meta I-JEPA and V-JEPA
//! examples: https://github.com/facebookresearch/ijepa and
//! https://github.com/facebookresearch/jepa. The Rust logo is only a compact,
//! redistributable input that keeps this example quick to run.

#[cfg(feature = "native")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use candlelighter::jepa::{BlockMask, JepaConfig, JepaGeometry, JepaTrainer};
    use image::ImageReader;
    use std::io::{Cursor, Read};

    const IMAGE_URL: &str = "https://www.rust-lang.org/logos/rust-logo-512x512.png";
    let response = ureq::get(IMAGE_URL).call()?;
    let mut bytes = Vec::new();
    response.into_reader().read_to_end(&mut bytes)?;
    let source = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()?
        .decode()?
        .resize_exact(32, 32, image::imageops::FilterType::Triangle)
        .to_luma8();
    let pixels = source
        .as_raw()
        .iter()
        .map(|pixel| *pixel as f32 / 255.0)
        .collect::<Vec<_>>();

    let geometry = JepaGeometry {
        height: 32,
        width: 32,
        channels: 1,
        patch_height: 8,
        patch_width: 8,
    };
    // Mask two spatially separated blocks, as I-JEPA predicts several target
    // regions from the same visible context.
    let masks = [
        BlockMask {
            row: 0,
            col: 0,
            height: 1,
            width: 2,
        },
        BlockMask {
            row: 2,
            col: 2,
            height: 2,
            width: 1,
        },
    ];
    let config = JepaConfig {
        patch_elements: 64,
        latent_dim: 32,
        learning_rate: 0.02,
        target_momentum: 0.996,
    };
    let mut ijepa = JepaTrainer::new(config, 23)?;
    for epoch in 0..20 {
        let loss = ijepa.train_image(&pixels, geometry, &masks)?;
        if epoch % 5 == 0 || epoch == 19 {
            println!("I-JEPA epoch {epoch:2}: loss={loss:.6}");
        }
    }

    // Turn the downloaded image into a two-frame clip. Repeating the same
    // spatial mask over both frames exercises V-JEPA tube masking.
    let mirrored = source
        .rows()
        .flat_map(|row| row.rev().map(|pixel| pixel[0] as f32 / 255.0))
        .collect::<Vec<_>>();
    let video = [pixels, mirrored].concat();
    let mut vjepa = JepaTrainer::new(config, 29)?;
    for epoch in 0..20 {
        let loss = vjepa.train_video(&video, 2, geometry, &masks)?;
        if epoch % 5 == 0 || epoch == 19 {
            println!("V-JEPA epoch {epoch:2}: loss={loss:.6}");
        }
    }
    Ok(())
}

#[cfg(not(feature = "native"))]
fn main() {
    eprintln!("enable this internet-backed example with --no-default-features --features native");
}
