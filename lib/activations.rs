
#[allow(unused)]
use crate::prelude::*;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Activations {
    Linear,
    Silu,
    Relu,
    Sigmoid,
    Softmax,
}

impl StringConstruction for Activations {
    fn from_string(name: String) -> Self {
        if name.to_lowercase().eq("linear"){
            return Activations::Linear;
        }
        else if name.to_lowercase().eq("silu"){
            return Activations::Silu;
        }
        else if name.to_lowercase().eq("relu"){
            return Activations::Relu;
        }
        else if name.to_lowercase().eq("sigmoid"){
            return Activations::Sigmoid;
        }
        else if name.to_lowercase().eq("softmax"){
            return Activations::Softmax;
        }
        panic!("Unknown activation")
    }
}