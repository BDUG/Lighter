use crate::prelude::*;

pub trait LayerTrait {
    fn new(perceptrons: usize, previousperceptrons: usize, activation: Activations, device: &Device, varmap : &VarMap, name: String) -> Self;
    fn typ(&self) -> String;
}
pub struct Dense {
    pub activation: Activations,
    pub perceptrons: usize,
    pub previousperceptrons: usize,
    pub denselayer: Linear,
    pub device: Device,
    pub name: String,
}

impl LayerTrait for Dense {
    fn new(perceptrons: usize, previousperceptrons: usize, activation: Activations, device: &Device, varmap : &VarMap, name: String) -> Self {
        let vs = VarBuilder::from_varmap(varmap, DType::F32, &device);
        let tmp_name = name.clone();
        Self {
            activation : activation,
            perceptrons : perceptrons,
            previousperceptrons : previousperceptrons,
            denselayer : linear(previousperceptrons, perceptrons,vs.pp(name)).unwrap(),
            device : device.clone(),
            name: tmp_name.clone(),
        }
    }

    fn typ(&self) -> String {
        "Dense".into()
    }
}


impl Dense {
    pub fn forward(&self, input: Tensor) -> Tensor {
        // Apply layer calculation
        let fullyconnected = self.denselayer.forward(&input);
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

/*
impl Clone for Dense {
    fn clone(&self) -> Dense {        
        let result = Dense::new(
            self.perceptrons, 
            self.previousperceptrons, 
            self.activation.clone(), 
            &self.device);
        return result;
    }
}  */