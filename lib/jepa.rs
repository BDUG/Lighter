//! Small, backend-independent I-JEPA and V-JEPA building blocks.
//!
//! The implementations intentionally operate on `f32` slices so they can be
//! used for experiments without selecting a tensor backend.  An application
//! supplies its encoder; [`MeanEncoder`] is a useful executable reference.

use std::fmt;

/// Errors returned by JEPA input validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JepaError(pub String);

impl fmt::Display for JepaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for JepaError {}

/// Maps one spatial patch to a latent representation.
pub trait PatchEncoder {
    fn latent_dim(&self) -> usize;
    fn encode(&self, patch: &[f32]) -> Result<Vec<f32>, JepaError>;
}

/// Reference encoder which returns chunk means (not a production backbone).
#[derive(Debug, Clone)]
pub struct MeanEncoder {
    latent_dim: usize,
}

impl MeanEncoder {
    pub fn new(latent_dim: usize) -> Result<Self, JepaError> {
        if latent_dim == 0 {
            return Err(JepaError("latent_dim must be positive".into()));
        }
        Ok(Self { latent_dim })
    }
}

impl PatchEncoder for MeanEncoder {
    fn latent_dim(&self) -> usize {
        self.latent_dim
    }
    fn encode(&self, patch: &[f32]) -> Result<Vec<f32>, JepaError> {
        if patch.is_empty() {
            return Err(JepaError("cannot encode an empty patch".into()));
        }
        let mut out = vec![0.0; self.latent_dim];
        let mut counts = vec![0usize; self.latent_dim];
        for (i, value) in patch.iter().enumerate() {
            out[i % self.latent_dim] += value;
            counts[i % self.latent_dim] += 1;
        }
        for (value, count) in out.iter_mut().zip(counts) {
            if count != 0 {
                *value /= count as f32;
            }
        }
        Ok(out)
    }
}

/// Rectangular region in patch coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockMask {
    pub row: usize,
    pub col: usize,
    pub height: usize,
    pub width: usize,
}

impl BlockMask {
    fn contains(&self, row: usize, col: usize) -> bool {
        row >= self.row
            && row < self.row + self.height
            && col >= self.col
            && col < self.col + self.width
    }
}

/// Output of a JEPA forward pass.
#[derive(Debug, Clone, PartialEq)]
pub struct JepaOutput {
    pub context: Vec<f32>,
    pub targets: Vec<Vec<f32>>,
    pub predictions: Vec<Vec<f32>>,
    pub loss: f32,
}

/// Image JEPA: predicts target-patch representations from visible context.
#[derive(Debug, Clone)]
pub struct IJepa<E> {
    encoder: E,
    patch_height: usize,
    patch_width: usize,
}

impl<E: PatchEncoder> IJepa<E> {
    pub fn new(encoder: E, patch_height: usize, patch_width: usize) -> Result<Self, JepaError> {
        if patch_height == 0 || patch_width == 0 {
            return Err(JepaError("patch dimensions must be positive".into()));
        }
        Ok(Self {
            encoder,
            patch_height,
            patch_width,
        })
    }

    /// Runs a forward pass over a channel-last image. Predictions use the mean
    /// context embedding; callers can replace this baseline with a trainable predictor.
    pub fn forward(
        &self,
        image: &[f32],
        height: usize,
        width: usize,
        channels: usize,
        targets: &[BlockMask],
    ) -> Result<JepaOutput, JepaError> {
        if height / self.patch_height * self.patch_height != height
            || width / self.patch_width * self.patch_width != width
            || channels == 0
        {
            return Err(JepaError(
                "image dimensions must be non-zero multiples of patch dimensions".into(),
            ));
        }
        if image.len() != height * width * channels {
            return Err(JepaError("image buffer has the wrong length".into()));
        }
        let rows = height / self.patch_height;
        let cols = width / self.patch_width;
        validate_masks(targets, rows, cols)?;
        let mut context_embeddings = Vec::new();
        let mut target_embeddings = Vec::new();
        for row in 0..rows {
            for col in 0..cols {
                let embedding = self.encoder.encode(&extract_patch(
                    image,
                    width,
                    channels,
                    row,
                    col,
                    self.patch_height,
                    self.patch_width,
                ))?;
                if targets.iter().any(|mask| mask.contains(row, col)) {
                    target_embeddings.push(embedding);
                } else {
                    context_embeddings.push(embedding);
                }
            }
        }
        finish(
            context_embeddings,
            target_embeddings,
            self.encoder.latent_dim(),
        )
    }
}

/// Video JEPA. A spatial target block is masked for every frame (a tube mask).
#[derive(Debug, Clone)]
pub struct VJepa<E> {
    image: IJepa<E>,
}

impl<E: PatchEncoder> VJepa<E> {
    pub fn new(encoder: E, patch_height: usize, patch_width: usize) -> Result<Self, JepaError> {
        Ok(Self {
            image: IJepa::new(encoder, patch_height, patch_width)?,
        })
    }

    /// Runs V-JEPA on frame-major, channel-last video data.
    pub fn forward(
        &self,
        video: &[f32],
        frames: usize,
        height: usize,
        width: usize,
        channels: usize,
        tube_masks: &[BlockMask],
    ) -> Result<JepaOutput, JepaError> {
        let frame_len = height
            .checked_mul(width)
            .and_then(|v| v.checked_mul(channels))
            .ok_or_else(|| JepaError("video dimensions overflow".into()))?;
        if frames == 0 || video.len() != frames * frame_len {
            return Err(JepaError("video buffer has the wrong length".into()));
        }
        let mut contexts = Vec::new();
        let mut targets = Vec::new();
        for frame in video.chunks_exact(frame_len) {
            let output = self
                .image
                .forward(frame, height, width, channels, tube_masks)?;
            contexts.push(output.context);
            targets.extend(output.targets);
        }
        finish(contexts, targets, self.image.encoder.latent_dim())
    }
}

fn finish(
    contexts: Vec<Vec<f32>>,
    targets: Vec<Vec<f32>>,
    dim: usize,
) -> Result<JepaOutput, JepaError> {
    if contexts.is_empty() {
        return Err(JepaError("the mask leaves no context patches".into()));
    }
    if targets.is_empty() {
        return Err(JepaError("at least one target patch is required".into()));
    }
    let mut context = vec![0.0; dim];
    for embedding in &contexts {
        for (sum, value) in context.iter_mut().zip(embedding) {
            *sum += value;
        }
    }
    for value in &mut context {
        *value /= contexts.len() as f32;
    }
    let predictions = vec![context.clone(); targets.len()];
    let count = targets.len() * dim;
    let loss = targets
        .iter()
        .zip(&predictions)
        .flat_map(|(a, b)| a.iter().zip(b))
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>()
        / count as f32;
    Ok(JepaOutput {
        context,
        targets,
        predictions,
        loss,
    })
}

fn validate_masks(masks: &[BlockMask], rows: usize, cols: usize) -> Result<(), JepaError> {
    for mask in masks {
        if mask.height == 0
            || mask.width == 0
            || mask.row + mask.height > rows
            || mask.col + mask.width > cols
        {
            return Err(JepaError(
                "target mask is empty or outside the patch grid".into(),
            ));
        }
    }
    Ok(())
}

fn extract_patch(
    image: &[f32],
    width: usize,
    channels: usize,
    patch_row: usize,
    patch_col: usize,
    ph: usize,
    pw: usize,
) -> Vec<f32> {
    let mut patch = Vec::with_capacity(ph * pw * channels);
    for row in patch_row * ph..(patch_row + 1) * ph {
        let start = (row * width + patch_col * pw) * channels;
        patch.extend_from_slice(&image[start..start + pw * channels]);
    }
    patch
}

/// Hyperparameters for the trainable JEPA reference model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JepaConfig {
    pub patch_elements: usize,
    pub latent_dim: usize,
    pub learning_rate: f32,
    /// Exponential-moving-average coefficient for the target encoder.
    pub target_momentum: f32,
}

/// Image layout and patch geometry shared by JEPA training calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JepaGeometry {
    pub height: usize,
    pub width: usize,
    pub channels: usize,
    pub patch_height: usize,
    pub patch_width: usize,
}

impl JepaConfig {
    fn validate(self) -> Result<Self, JepaError> {
        if self.patch_elements == 0 || self.latent_dim == 0 {
            return Err(JepaError("JEPA dimensions must be positive".into()));
        }
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err(JepaError(
                "learning_rate must be positive and finite".into(),
            ));
        }
        if !self.target_momentum.is_finite()
            || self.target_momentum < 0.0
            || self.target_momentum >= 1.0
        {
            return Err(JepaError("target_momentum must be in [0, 1)".into()));
        }
        Ok(self)
    }
}

/// A trainable JEPA with an online encoder, stop-gradient EMA target encoder,
/// and position-conditioned predictor.
///
/// This is a fully executable linear reference implementation of the I-JEPA
/// and V-JEPA training algorithm. Production users can retain its masking and
/// training semantics while replacing the linear maps with an autograd-backed
/// ViT.
#[derive(Debug, Clone)]
pub struct JepaTrainer {
    config: JepaConfig,
    online_weights: Vec<f32>,
    online_bias: Vec<f32>,
    target_weights: Vec<f32>,
    target_bias: Vec<f32>,
    // Predictor input is context latent + normalized (time, row, column).
    predictor_weights: Vec<f32>,
    predictor_bias: Vec<f32>,
    steps: u64,
}

impl JepaTrainer {
    pub fn new(config: JepaConfig, seed: u64) -> Result<Self, JepaError> {
        let config = config.validate()?;
        let mut rng = SeededRng(seed);
        let scale = (2.0 / (config.patch_elements + config.latent_dim) as f32).sqrt();
        let online_weights = (0..config.latent_dim * config.patch_elements)
            .map(|_| rng.symmetric() * scale)
            .collect::<Vec<_>>();
        let predictor_weights = (0..config.latent_dim * (config.latent_dim + 3))
            .map(|_| rng.symmetric() * scale)
            .collect();
        Ok(Self {
            config,
            target_weights: online_weights.clone(),
            online_weights,
            online_bias: vec![0.0; config.latent_dim],
            target_bias: vec![0.0; config.latent_dim],
            predictor_weights,
            predictor_bias: vec![0.0; config.latent_dim],
            steps: 0,
        })
    }

    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// One I-JEPA optimization step on a channel-last image.
    pub fn train_image(
        &mut self,
        image: &[f32],
        geometry: JepaGeometry,
        masks: &[BlockMask],
    ) -> Result<f32, JepaError> {
        let samples = make_samples(image, 1, geometry, masks)?;
        self.train_samples(&samples)
    }

    /// One V-JEPA optimization step. Spatial masks are repeated through time,
    /// producing the tube masks used by V-JEPA.
    pub fn train_video(
        &mut self,
        video: &[f32],
        frames: usize,
        geometry: JepaGeometry,
        tube_masks: &[BlockMask],
    ) -> Result<f32, JepaError> {
        let samples = make_samples(video, frames, geometry, tube_masks)?;
        self.train_samples(&samples)
    }

    fn train_samples(&mut self, samples: &[PatchSample]) -> Result<f32, JepaError> {
        if samples
            .iter()
            .any(|sample| sample.values.len() != self.config.patch_elements)
        {
            return Err(JepaError(
                "patch size does not match JepaConfig::patch_elements".into(),
            ));
        }
        let contexts = samples.iter().filter(|s| !s.target).collect::<Vec<_>>();
        let targets = samples.iter().filter(|s| s.target).collect::<Vec<_>>();
        if contexts.is_empty() || targets.is_empty() {
            return Err(JepaError("a batch needs context and target patches".into()));
        }
        let dim = self.config.latent_dim;
        let mut context = vec![0.0; dim];
        for sample in &contexts {
            let encoded = linear(&self.online_weights, &self.online_bias, &sample.values);
            for (sum, value) in context.iter_mut().zip(encoded) {
                *sum += value / contexts.len() as f32;
            }
        }

        let mut grad_predictor = vec![0.0; self.predictor_weights.len()];
        let mut grad_predictor_bias = vec![0.0; dim];
        let mut grad_context = vec![0.0; dim];
        let mut loss = 0.0;
        for sample in &targets {
            let mut predictor_input = context.clone();
            predictor_input.extend_from_slice(&sample.position);
            let prediction = linear(
                &self.predictor_weights,
                &self.predictor_bias,
                &predictor_input,
            );
            let target = linear(&self.target_weights, &self.target_bias, &sample.values);
            for output in 0..dim {
                let error = prediction[output] - target[output];
                loss += error * error;
                let gradient = 2.0 * error / (targets.len() * dim) as f32;
                grad_predictor_bias[output] += gradient;
                let offset = output * predictor_input.len();
                for input in 0..predictor_input.len() {
                    grad_predictor[offset + input] += gradient * predictor_input[input];
                    if input < dim {
                        grad_context[input] += gradient * self.predictor_weights[offset + input];
                    }
                }
            }
        }
        loss /= (targets.len() * dim) as f32;

        let mut grad_online = vec![0.0; self.online_weights.len()];
        let mut grad_online_bias = vec![0.0; dim];
        for sample in &contexts {
            for output in 0..dim {
                let gradient = grad_context[output] / contexts.len() as f32;
                grad_online_bias[output] += gradient;
                for input in 0..self.config.patch_elements {
                    grad_online[output * self.config.patch_elements + input] +=
                        gradient * sample.values[input];
                }
            }
        }
        descend(
            &mut self.predictor_weights,
            &grad_predictor,
            self.config.learning_rate,
        );
        descend(
            &mut self.predictor_bias,
            &grad_predictor_bias,
            self.config.learning_rate,
        );
        descend(
            &mut self.online_weights,
            &grad_online,
            self.config.learning_rate,
        );
        descend(
            &mut self.online_bias,
            &grad_online_bias,
            self.config.learning_rate,
        );
        ema(
            &mut self.target_weights,
            &self.online_weights,
            self.config.target_momentum,
        );
        ema(
            &mut self.target_bias,
            &self.online_bias,
            self.config.target_momentum,
        );
        self.steps += 1;
        Ok(loss)
    }
}

#[derive(Debug)]
struct PatchSample {
    values: Vec<f32>,
    position: [f32; 3],
    target: bool,
}

fn make_samples(
    data: &[f32],
    frames: usize,
    geometry: JepaGeometry,
    masks: &[BlockMask],
) -> Result<Vec<PatchSample>, JepaError> {
    let JepaGeometry {
        height,
        width,
        channels,
        patch_height: ph,
        patch_width: pw,
    } = geometry;
    if frames == 0
        || ph == 0
        || pw == 0
        || channels == 0
        || height / ph * ph != height
        || width / pw * pw != width
    {
        return Err(JepaError("invalid video or patch dimensions".into()));
    }
    let frame_len = height
        .checked_mul(width)
        .and_then(|x| x.checked_mul(channels))
        .ok_or_else(|| JepaError("dimensions overflow".into()))?;
    if data.len()
        != frame_len
            .checked_mul(frames)
            .ok_or_else(|| JepaError("dimensions overflow".into()))?
    {
        return Err(JepaError("input buffer has the wrong length".into()));
    }
    let rows = height / ph;
    let cols = width / pw;
    validate_masks(masks, rows, cols)?;
    let mut samples = Vec::with_capacity(frames * rows * cols);
    for (time, frame) in data.chunks_exact(frame_len).enumerate() {
        for row in 0..rows {
            for col in 0..cols {
                samples.push(PatchSample {
                    values: extract_patch(frame, width, channels, row, col, ph, pw),
                    position: [
                        normalize(time, frames),
                        normalize(row, rows),
                        normalize(col, cols),
                    ],
                    target: masks.iter().any(|mask| mask.contains(row, col)),
                });
            }
        }
    }
    Ok(samples)
}

fn normalize(value: usize, length: usize) -> f32 {
    if length <= 1 {
        0.0
    } else {
        value as f32 / (length - 1) as f32 * 2.0 - 1.0
    }
}
fn linear(weights: &[f32], bias: &[f32], input: &[f32]) -> Vec<f32> {
    bias.iter()
        .enumerate()
        .map(|(row, bias)| {
            *bias
                + weights[row * input.len()..(row + 1) * input.len()]
                    .iter()
                    .zip(input)
                    .map(|(w, x)| w * x)
                    .sum::<f32>()
        })
        .collect()
}
fn descend(values: &mut [f32], gradients: &[f32], rate: f32) {
    for (value, gradient) in values.iter_mut().zip(gradients) {
        *value -= rate * gradient;
    }
}
fn ema(target: &mut [f32], online: &[f32], momentum: f32) {
    for (target, online) in target.iter_mut().zip(online) {
        *target = momentum * *target + (1.0 - momentum) * online;
    }
}

struct SeededRng(u64);
impl SeededRng {
    fn symmetric(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn image_and_video_jepa_execute() {
        let mask = [BlockMask {
            row: 0,
            col: 0,
            height: 1,
            width: 1,
        }];
        let image = IJepa::new(MeanEncoder::new(2).unwrap(), 2, 2).unwrap();
        let output = image
            .forward(
                &(0..16).map(|x| x as f32).collect::<Vec<_>>(),
                4,
                4,
                1,
                &mask,
            )
            .unwrap();
        assert_eq!(output.targets.len(), 1);
        assert!(output.loss.is_finite());
        let video = VJepa::new(MeanEncoder::new(2).unwrap(), 2, 2).unwrap();
        assert_eq!(
            video
                .forward(&vec![1.0; 32], 2, 4, 4, 1, &mask)
                .unwrap()
                .targets
                .len(),
            2
        );
    }

    #[test]
    fn trainable_jepa_updates_image_and_video() {
        let config = JepaConfig {
            patch_elements: 4,
            latent_dim: 3,
            learning_rate: 0.01,
            target_momentum: 0.99,
        };
        let mut model = JepaTrainer::new(config, 7).unwrap();
        let geometry = JepaGeometry {
            height: 4,
            width: 4,
            channels: 1,
            patch_height: 2,
            patch_width: 2,
        };
        let mask = [BlockMask {
            row: 0,
            col: 0,
            height: 1,
            width: 1,
        }];
        let image = (0..16).map(|x| x as f32 / 16.0).collect::<Vec<_>>();
        assert!(model
            .train_image(&image, geometry, &mask)
            .unwrap()
            .is_finite());
        let video = [image.clone(), image].concat();
        assert!(model
            .train_video(&video, 2, geometry, &mask)
            .unwrap()
            .is_finite());
        assert_eq!(model.steps(), 2);
    }
}
