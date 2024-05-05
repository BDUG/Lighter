
#[allow(unused)]
use crate::prelude::*;


#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SaveWeightsType {
    SafeTensor,
    PlainJSON,
    HuggingfaceHub,
}


impl StringConstruction for SaveWeightsType {
    fn from_string(name: String) -> Self {
        if name.to_lowercase().eq("safetensor"){
            return SaveWeightsType::SafeTensor;
        }
        else if name.to_lowercase().eq("json"){
            return SaveWeightsType::PlainJSON;
        }
        else if name.to_lowercase().eq("huggingfacehub"){
            return SaveWeightsType::HuggingfaceHub;
        }
        panic!("Unknown optimizer {}",name)
    }
}