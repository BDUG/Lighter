
#[allow(unused)]
pub use crate::prelude::*;

pub struct Flatten {
    pub device: Device,
    pub name: String,
}


pub trait FlattenLayerTrait {
    fn new(device: &Device, varmap : &VarMap, name: String) -> Self;
}


impl FlattenLayerTrait for Flatten {
    fn new(device: &Device, varmap : &VarMap, name: String) -> Self {
        let tmp_name = name.clone();
        Self {
            device : device.clone(),
            name: tmp_name.clone(),
        }
    }

}


impl Trainable for Flatten {

    fn forward(&self, input: Tensor) -> Tensor {
        let mut result = input.flatten_all().unwrap();
        result = result.reshape( (1, result.elem_count())).unwrap();
        return result;
    }

    fn typ(&self) -> String {
        "Flatten".into()
    }

    fn input_perceptrons(&self) -> u32{
        return 1;
    }
    fn output_perceptrons(&self) -> u32{
        return 1;
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
    
}
