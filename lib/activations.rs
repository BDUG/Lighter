#[allow(unused)]
use crate::prelude::*;

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub enum Activations {
    Linear,
    Silu,
    Relu,
    Sigmoid,
    Softmax,
}