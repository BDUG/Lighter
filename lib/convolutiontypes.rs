#[allow(unused)]
use crate::prelude::*;

// Also see 
// https://github.com/huggingface/candle/blob/main/candle-nn/src/init.rs#L105
// https://en.wikipedia.org/wiki/Kernel_(image_processing)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConvolutionTypes {
    None,
    Default, // what is equal to random
    Identity,
    LaPlace, 
    Edge,
    Sharpen,
    KaimingReluNormal,
    KaimingReluUniform,
    KaimingLinearNormal,
    KaimingLinearUniform,
    KaimingSigmoidNormal,
    KaimingSigmoidUniform,
    KaimingTanhNormal,
    KaimingTanhUniform,
    KaimingSeluNormal,
    KaimingSeluUniform,
}


impl StringConstruction for ConvolutionTypes {
    fn from_string(name: String) -> Self {
        if name.to_lowercase().eq("default"){
            return ConvolutionTypes::Default;
        }
        else if name.to_lowercase().eq("identity"){
            return ConvolutionTypes::Identity;
        }
        else if name.to_lowercase().eq("laplace"){
            return ConvolutionTypes::LaPlace;
        }
        else if name.to_lowercase().eq("edge"){
            return ConvolutionTypes::Edge;
        }
        else if name.to_lowercase().eq("sharpen"){
            return ConvolutionTypes::Sharpen;
        }
        else if name.to_lowercase().eq("kaimingrelunormal"){
            return ConvolutionTypes::KaimingReluNormal;
        }
        else if name.to_lowercase().eq("kaimingreluuniform"){
            return ConvolutionTypes::KaimingReluUniform;
        }
        else if name.to_lowercase().eq("kaiminglinearnormal"){
            return ConvolutionTypes::KaimingLinearNormal;
        }
        else if name.to_lowercase().eq("kaiminglinearuniform"){
            return ConvolutionTypes::KaimingLinearUniform;
        }
        else if name.to_lowercase().eq("kaimingsigmoidnormal"){
            return ConvolutionTypes::KaimingSigmoidNormal;
        }
        else if name.to_lowercase().eq("kaimingsigmoiduniform"){
            return ConvolutionTypes::KaimingSigmoidUniform;
        }
        else if name.to_lowercase().eq("kaimingtanhnormal"){
            return ConvolutionTypes::KaimingTanhNormal;
        }
        else if name.to_lowercase().eq("kaimingtanhuniform"){
            return ConvolutionTypes::KaimingTanhUniform;
        }
        else if name.to_lowercase().eq("kaimingselunormal"){
            return ConvolutionTypes::KaimingSeluNormal;
        }
        else if name.to_lowercase().eq("kaimingseluuniform"){
            return ConvolutionTypes::KaimingSeluUniform;
        }
        else if name.to_lowercase().eq("none"){
            return ConvolutionTypes::None;
        }
        panic!("Unknown convolution type")
    }
}