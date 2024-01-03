

use serde::ser::SerializeStructVariant;

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

}

pub trait Trainable {
    fn forward(&self, input: Tensor) -> Tensor;
    fn typ(&self) -> String;
    fn inputPerceptrons(&self) -> u32;
    fn outputPerceptrons(&self) -> u32;
    fn as_any(&self) -> &dyn Any;
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

    fn typ(&self) -> String {
        "Conv".into()
    }

    fn inputPerceptrons(&self) -> u32{
        return 1;
    }
    fn outputPerceptrons(&self) -> u32{
        return 1;
    }

    fn as_any(&self) -> &dyn Any {
        self
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

    fn typ(&self) -> String {
        "Dense".into()
    }

    fn inputPerceptrons(&self) -> u32{
        return self.previousperceptrons as u32;
    }
    fn outputPerceptrons(&self) -> u32{
        return self.perceptrons as u32;
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
    
}


impl Serialize for dyn Trainable {

    fn serialize<S>(&self, serializer: S) -> std::result::Result<<S as Serializer>::Ok, <S as Serializer>::Error>
    where
        S: Serializer,
    {
        if self.typ().eq("Dense") {
            let mut state = serializer.serialize_struct("Dense", 4)?;
            // One of two ways to downcast in Rust
            let dense: &Dense = match self.as_any().downcast_ref::<Dense>() {
                Some(b) => b,
                None => panic!("Not a Dense type"),
            };
            state.serialize_field("perceptrons", &dense.perceptrons)?;
            state.serialize_field("previousperceptrons", &dense.previousperceptrons)?;
            state.serialize_field("activation", &dense.activation)?;
            state.serialize_field("name", &dense.name)?;
            return Ok(state.end().unwrap());
        }
        else if self.typ().eq("Conv") {
            let mut state = serializer.serialize_struct("Conv", 5)?;
            // One of two ways to downcast in Rust
            let conv: &Conv = match self.as_any().downcast_ref::<Conv>() {
                Some(b) => b,
                None => panic!("Not a Dense type"),
            };
            // TODO
            //state.serialize_field("kernel", &conv.kernel)?;
            state.serialize_field("dimensionality", &conv.dimensionality)?;
            state.serialize_field("padding", &conv.padding)?;
            state.serialize_field("stride", &conv.stride)?;
            state.serialize_field("dilation", &conv.dilation)?;
            state.serialize_field("groups", &conv.groups)?;
            state.serialize_field("name", &conv.name)?;
            return Ok(state.end().unwrap());
        }
        panic!("Unknown layer type")
    }
}
