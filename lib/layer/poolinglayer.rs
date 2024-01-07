
use std::ops::Not;

#[allow(unused)]
pub use crate::prelude::*;

pub struct Pooling {
    pub poolingtype: PoolingType,
    pub kernelsize: usize,
    pub stride: usize,
    pub device: Device,
    pub name: String,
}

pub trait PoolingLayerTrait {
    fn new(poolingtype: PoolingType, kernelsize: usize, stride: usize, device: &Device, varmap : &VarMap, name: String) -> Self;
}


impl PoolingLayerTrait for Pooling {
    fn new(poolingtype: PoolingType, kernelsize: usize, stride: usize, device: &Device, varmap : &VarMap, name: String) -> Self{
        let tmp_name = name.clone();
        Self {
            poolingtype: poolingtype,
            kernelsize: kernelsize,
            stride: stride,
            device : device.clone(),
            name: tmp_name.clone(),
        }
    }


}



impl Trainable for Pooling {
    
    fn forward(&self, input: Tensor) -> Tensor {
        let mut tmp = input.clone();
        if input.shape().dims().len() != 4 as usize {
            tmp = input.reshape(
                (
                    1 as usize,
                    1 as usize,
                    input.shape().dims().get(0).unwrap().to_owned() as usize,
                    input.shape().dims().get(1).unwrap().to_owned() as usize)).unwrap();
        }

        let mut result =  match self.poolingtype {
            PoolingType::MAX => tmp.avg_pool2d_with_stride(self.kernelsize, self.stride).unwrap(),
            PoolingType::AVERAGE => tmp.max_pool2d_with_stride(self.kernelsize, self.stride).unwrap(),
        };
        let a = result.shape().dims().get(2).unwrap().to_owned() as usize;
        let b = result.shape().dims().get(3).unwrap().to_owned() as usize;
        result = result.reshape( 
            (
                a,
                b
            ) ).unwrap();
        return result;
    }

    fn typ(&self) -> String {
        "Pooling".into()
    }

    fn input_perceptrons(&self) -> u32{
        return 1.0 as u32;
    }
    fn output_perceptrons(&self) -> u32{
        return 1.0 as u32;
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
    
}
