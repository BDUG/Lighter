#[allow(unused)]
pub use crate::prelude::*;

pub struct Conv {
    pub kernel: Tensor,
    pub dimensionality: usize,
    pub padding: usize,
    pub stride: usize,
    pub dilation: usize,
    pub groups: usize,
    pub device: Device,
    pub name: String,
}


pub trait ConvLayerTrait {
    fn new(kernel: Tensor, dimensionality: usize, padding: usize, stride: usize, dilation: usize, groups: usize, device: &Device, varmap : &VarMap, name: String) -> Self;
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
