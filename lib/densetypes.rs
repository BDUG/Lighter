#[allow(unused)]
use crate::prelude::*;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DenseType {
    Standard,
    LORA,
    DORA
}


impl StringConstruction for DenseType {
    fn from_string(name: String) -> Self {
        if name.to_lowercase().eq("standard"){
            return DenseType::Standard;
        }
        else if name.to_lowercase().eq("lora"){
            return DenseType::LORA;
        }
        else if name.to_lowercase().eq("dora"){
            return DenseType::DORA;
        }
        panic!("Unknown dense type")
    }
}