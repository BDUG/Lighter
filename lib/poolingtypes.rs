#[allow(unused)]
use crate::prelude::*;


#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PoolingType {
    MAX,
    AVERAGE,
}


impl StringConstruction for PoolingType {
    fn from_string(name: String) -> Self {
        if name.to_lowercase().eq("max"){
            return PoolingType::MAX;
        }
        else if name.to_lowercase().eq("average"){
            return PoolingType::AVERAGE;
        }
        panic!("Unknown pooling type")
    }
}