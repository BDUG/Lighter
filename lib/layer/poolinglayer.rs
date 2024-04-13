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
    fn new(poolingtype: PoolingType, kernelsize: usize, stride: usize, device: &Device, _varmap : &VarMap, name: String) -> Self{
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
        let timesteps = *input.shape().dims().get(0).unwrap();
        let height = *input.shape().dims().get(1).unwrap();
        let weight = *input.shape().dims().get(2).unwrap();
        tmp = tmp.reshape( (1,timesteps,height,weight) ).unwrap();

        let result =  match self.poolingtype {
            PoolingType::MAX => tmp.avg_pool2d_with_stride(self.kernelsize, self.stride).unwrap(),
            PoolingType::AVERAGE => tmp.max_pool2d_with_stride(self.kernelsize, self.stride).unwrap(),
        };
        let height = *result.shape().dims().get(2).unwrap();
        let weight = *result.shape().dims().get(3).unwrap();
        return result.reshape( (timesteps,height,weight) ).unwrap();
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
