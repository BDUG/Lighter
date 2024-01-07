

#[allow(unused)]
pub use crate::prelude::*;

pub struct Dense {
    pub activation: Activations,
    pub perceptrons: usize,
    pub previousperceptrons: usize,
    pub denselayer: Linear,
    pub device: Device,
    pub name: String,
}

pub trait DenseLayerTrait {
    fn new(perceptrons: usize, previousperceptrons: usize, activation: Activations, device: &Device, varmap : &VarMap, name: String) -> Self;
}

impl DenseLayerTrait for Dense {
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

}

impl Trainable for Dense {
    
    // .to_scalar::<f64>()
    fn forward(&self, input: Tensor) -> Tensor {
        // Apply layer calculation
        let new_tensor = input.to_dtype(DType::F32).unwrap();
        let fullyconnected = self.denselayer.forward(&new_tensor);
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

    fn typ(&self) -> String {
        "Dense".into()
    }

    fn input_perceptrons(&self) -> u32{
        return self.previousperceptrons as u32;
    }
    fn output_perceptrons(&self) -> u32{
        return self.perceptrons as u32;
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
    
}

