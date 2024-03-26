#[allow(unused)]
pub use crate::prelude::*;

pub struct Normalization {
    pub axis: u64,
    pub device: Device,
    pub name: String,
}


pub trait NormalizationLayerTrait {
    fn new(axis: u64, device: &Device, varmap : &VarMap, name: String) -> Self;
}


impl NormalizationLayerTrait for Normalization {
    fn new(axis: u64, device: &Device, _varmap : &VarMap, name: String) -> Self {
        let tmp_name = name.clone();
        Self {
            axis: axis,
            device : device.clone(),
            name: tmp_name.clone(),
        }
    }

}


impl Trainable for Normalization {

    fn forward(&self, input: Tensor) -> Tensor {
        let result = input.clone();
        let _ = result.normalize_axis(self.axis as i64);
        return result;
    }

    fn typ(&self) -> String {
        "Normalization".into()
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
