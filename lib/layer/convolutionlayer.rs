#[allow(unused)]
pub use crate::prelude::*;

pub struct Conv {
    pub kernel: Tensor,
    pub kerneltype: ConvolutionTypes,
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
    fn new2(kerneltype: ConvolutionTypes, dimensionality: usize, padding: usize, stride: usize, dilation: usize, groups: usize, device: &Device, varmap : &VarMap, name: String) -> Self;
}


impl ConvLayerTrait for Conv {
    fn new(kernel: Tensor, dimensionality: usize, padding: usize, stride: usize, dilation: usize, groups: usize, device: &Device, varmap : &VarMap, name: String) -> Self {
        let tmp_name = name.clone();
        Self {
            kernel: kernel,
            kerneltype: ConvolutionTypes::None,
            dimensionality : dimensionality,
            padding : padding,
            stride : stride,
            dilation : dilation,
            groups : groups,
            device : device.clone(),
            name: tmp_name.clone(),
        }
    }

    fn new2(kerneltype: ConvolutionTypes, dimensionality: usize, padding: usize, stride: usize, dilation: usize, groups: usize, device: &Device, varmap : &VarMap, name: String) -> Self {
        let tmp_name = name.clone();
        Self {
            kernel: Tensor::new(&[[3f32, 0., 0.]], &device).unwrap(),
            kerneltype: kerneltype,
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
        if self.kerneltype.eq(&ConvolutionTypes::None) {
            if self.dimensionality.eq(&(1 as usize)){
                return input.conv1d(&self.kernel, self.padding, self.stride, self.dilation, self.groups).unwrap();
            }
            else if self.dimensionality.eq(&(2 as usize)){
                return input.conv2d(&self.kernel, self.padding, self.stride, self.dilation, self.groups).unwrap();
            }        
            panic!("Dimensionality not implemented");
        }
        else {
            let mut rtensor = Tensor::zeros(input.shape(), DType::F32, &self.device).unwrap();
            if self.kerneltype.eq(&ConvolutionTypes::Default) {
                // FIXME
                let mean= 1.0;
                let standard_deviation = 1.0;
                let kernelconfig = Init::Randn { mean: mean, stdev: standard_deviation };
                rtensor.clone_from(kernelconfig.var(input.shape(), DType::F64, &self.device).unwrap().as_tensor());
                return rtensor;
            }
            else if self.kerneltype.eq(&ConvolutionTypes::Identity){
                rtensor.clone_from(Init::Const(1.0).var(input.shape(), DType::F64, &self.device).unwrap().as_tensor());
                return rtensor;
            }
            else if self.kerneltype.eq(&ConvolutionTypes::KaimingLinearNormal){
                let kernelconfig = Init::Kaiming { dist:init::NormalOrUniform::Normal, fan: init::FanInOut::FanIn, non_linearity: init::NonLinearity::Linear };
                rtensor.clone_from(kernelconfig.var(input.shape(), DType::F64, &self.device).unwrap().as_tensor());
                return rtensor;
            }
            else if self.kerneltype.eq(&ConvolutionTypes::KaimingLinearUniform){
                let kernelconfig = Init::Kaiming { dist:init::NormalOrUniform::Uniform, fan: init::FanInOut::FanIn, non_linearity: init::NonLinearity::Linear };
                rtensor.clone_from(kernelconfig.var(input.shape(), DType::F64, &self.device).unwrap().as_tensor());
                return rtensor;
            }
            else if self.kerneltype.eq(&ConvolutionTypes::KaimingReluNormal){
                let kernelconfig = Init::Kaiming { dist:init::NormalOrUniform::Normal, fan: init::FanInOut::FanIn, non_linearity: init::NonLinearity::ReLU };
                rtensor.clone_from(kernelconfig.var(input.shape(), DType::F64, &self.device).unwrap().as_tensor());
                return rtensor;
            }
            else if self.kerneltype.eq(&ConvolutionTypes::KaimingReluUniform){
                let kernelconfig = Init::Kaiming { dist:init::NormalOrUniform::Uniform, fan: init::FanInOut::FanIn, non_linearity: init::NonLinearity::ReLU };
                rtensor.clone_from(kernelconfig.var(input.shape(), DType::F64, &self.device).unwrap().as_tensor());
                return rtensor;
            }
            else if self.kerneltype.eq(&ConvolutionTypes::KaimingSeluNormal){
                let kernelconfig = Init::Kaiming { dist:init::NormalOrUniform::Normal, fan: init::FanInOut::FanIn, non_linearity: init::NonLinearity::SELU };
                rtensor.clone_from(kernelconfig.var(input.shape(), DType::F64, &self.device).unwrap().as_tensor());
                return rtensor;
            }
            else if self.kerneltype.eq(&ConvolutionTypes::KaimingSeluUniform){
                let kernelconfig = Init::Kaiming { dist:init::NormalOrUniform::Uniform, fan: init::FanInOut::FanIn, non_linearity: init::NonLinearity::SELU };
                rtensor.clone_from(kernelconfig.var(input.shape(), DType::F64, &self.device).unwrap().as_tensor());
                return rtensor;
            }
            else if self.kerneltype.eq(&ConvolutionTypes::KaimingSigmoidNormal){
                let kernelconfig = Init::Kaiming { dist:init::NormalOrUniform::Normal, fan: init::FanInOut::FanIn, non_linearity: init::NonLinearity::Sigmoid };
                rtensor.clone_from(kernelconfig.var(input.shape(), DType::F64, &self.device).unwrap().as_tensor());
                return rtensor;
            }
            else if self.kerneltype.eq(&ConvolutionTypes::KaimingSigmoidUniform){
                let kernelconfig = Init::Kaiming { dist:init::NormalOrUniform::Uniform, fan: init::FanInOut::FanIn, non_linearity: init::NonLinearity::Sigmoid };
                rtensor.clone_from(kernelconfig.var(input.shape(), DType::F64, &self.device).unwrap().as_tensor());
                return rtensor;
            }
            else if self.kerneltype.eq(&ConvolutionTypes::KaimingTanhNormal){
                let kernelconfig = Init::Kaiming { dist:init::NormalOrUniform::Normal, fan: init::FanInOut::FanIn, non_linearity: init::NonLinearity::Tanh };
                rtensor.clone_from(kernelconfig.var(input.shape(), DType::F64, &self.device).unwrap().as_tensor());
                return rtensor;
            }
            else if self.kerneltype.eq(&ConvolutionTypes::KaimingTanhUniform) {
                let kernelconfig = Init::Kaiming { dist:init::NormalOrUniform::Uniform, fan: init::FanInOut::FanIn, non_linearity: init::NonLinearity::Tanh };
                rtensor.clone_from(kernelconfig.var(input.shape(), DType::F64, &self.device).unwrap().as_tensor());
                return rtensor;
            }
            else {
                panic!("Not implemented");
            }
        }
    }

    fn typ(&self) -> String {
        "Conv".into()
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
