
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
        let vs = VarBuilder::from_varmap(varmap, DType::F32, &device);
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
        let result =  match self.poolingtype {
            PoolingType::MAX => input.avg_pool2d_with_stride(self.kernelsize, self.stride).unwrap(),
            PoolingType::AVERAGE => input.max_pool2d_with_stride(self.kernelsize, self.stride).unwrap(),
        };
        return result;
    }

    fn typ(&self) -> String {
        "Pooling".into()
    }

    fn inputPerceptrons(&self) -> u32{
        return 1.0 as u32;
    }
    fn outputPerceptrons(&self) -> u32{
        return 1.0 as u32;
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
    
}
