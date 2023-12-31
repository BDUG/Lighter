use crate::prelude::*;
use serde::{Serialize, Deserialize, Deserializer};

pub trait LayerTrait {
    fn new(perceptron: usize, prev: usize, device: &Device, activation: Activations) -> Self;
    fn typ(&self) -> String;
}
pub struct Dense {
    pub activation: Activations,
    pub perceptrons: usize,
    pub previousperceptrons: usize,

    pub device: Device,
}

impl LayerTrait for Dense {
    fn new(perceptrons: usize, previousperceptrons: usize, device: &Device, activation: Activations) -> Self {
        Self {
            activation,
            perceptrons,
            previousperceptrons,
            device: device.clone(),
        }
    }

    fn typ(&self) -> String {
        "Dense".into()
    }
}


impl Dense {
    pub fn forward(&self, input: Tensor) -> Tensor {
        // Apply layer calculation
        let weights = Tensor::ones( (self.perceptrons,self.previousperceptrons), DType::F32, &self.device).unwrap();
        let fullyconnected = Linear::new(weights, None).forward(&input);
        let fullyconnected_checked = match fullyconnected {
            Ok(fullyconnected) => fullyconnected,
            Err(error) => panic!("{}",error.to_string()),
        };
        // Apply activation
        let activated = match self.activation {
            Activations::Linear => Ok(fullyconnected_checked.clone()),
            Activations::Relu => fullyconnected_checked.clone().relu(),
            Activations::Silu => ops::silu(&fullyconnected_checked.clone()),
            Activations::Sigmoid => ops::sigmoid(&fullyconnected_checked.clone()),
            Activations::Softmax => ops::log_softmax(&fullyconnected_checked.clone(), D::Minus1),
        };
        let activated_checked = match activated {
            Ok(activated) => activated,
            Err(error) => panic!("{}",error.to_string()),
        };
        return activated_checked;
    }

}

impl Clone for Dense {
    fn clone(&self) -> Dense {        
        let result = Dense::new(
            self.perceptrons, 
            self.previousperceptrons, 
            &self.device, 
            self.activation.clone());
        return result;
    }
}