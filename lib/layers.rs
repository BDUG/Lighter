

#[allow(unused)]
use crate::prelude::*;


pub struct Conv {
    kernel: Tensor,
    dimensionality: usize,
    padding: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
    pub device: Device,
    pub name: String,
}


pub struct Dense {
    pub activation: Activations,
    pub perceptrons: usize,
    pub previousperceptrons: usize,
    pub denselayer: Linear,
    pub device: Device,
    pub name: String,
}

pub trait ConvLayerTrait {
    fn new(kernel: Tensor, dimensionality: usize, padding: usize, stride: usize, dilation: usize, groups: usize, device: &Device, varmap : &VarMap, name: String) -> Self;
    fn typ(&self) -> String;
}

pub trait DenseLayerTrait {
    fn new(perceptrons: usize, previousperceptrons: usize, activation: Activations, device: &Device, varmap : &VarMap, name: String) -> Self;
    fn typ(&self) -> String;
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

    fn typ(&self) -> String {
        "Dense".into()
    }
}

impl ConvLayerTrait for Conv {
    fn new(kernel: Tensor, dimensionality: usize, padding: usize, stride: usize, dilation: usize, groups: usize, device: &Device, varmap : &VarMap, name: String) -> Self {
        let vs = VarBuilder::from_varmap(varmap, DType::F32, &device);
        let tmp_name = name.clone();
        Self {
            kernel: kernel,
            dimensionality : dimensionality,
            padding : padding,
            stride : stride,
            dilation : dilation,
            groups : groups,
            device : device.clone(),
            name: tmp_name.clone(),
        }
    }

    fn typ(&self) -> String {
        "Conv".into()
    }
}

pub trait Trainable {
    fn forward(&self, input: Tensor) -> Tensor;
}

impl Conv {

}

impl Trainable for Conv {

    fn forward(&self, input: Tensor) -> Tensor {
        if self.dimensionality.eq(&(1 as usize)){
            return input.conv1d(&self.kernel, self.padding, self.stride, self.dilation, self.groups).unwrap();
        }
        else if self.dimensionality.eq(&(2 as usize)){
            return input.conv2d(&self.kernel, self.padding, self.stride, self.dilation, self.groups).unwrap();
        }        
        panic!("Dimensionality not implemented");
    }
}

impl Dense {

}


impl Trainable for Dense {
    
    fn forward(&self, input: Tensor) -> Tensor {
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


impl Serialize for Dense {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<<S as Serializer>::Ok, <S as Serializer>::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("Dense", 4)?;

        state.serialize_field("perceptrons", &self.perceptrons)?;
        state.serialize_field("previousperceptrons", &self.previousperceptrons)?;
        state.serialize_field("activation", &self.activation)?;
        state.serialize_field("name", &self.name)?;

        return Ok(state.end().unwrap());
    }
}
