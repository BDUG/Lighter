#[allow(unused)]
use crate::prelude::*;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmbeddingType {
    Standard
}


impl StringConstruction for EmbeddingType {
    fn from_string(name: String) -> Self {
        if name.to_lowercase().eq("standard"){
            return EmbeddingType::Standard;
        }
        panic!("Unknown recurrent type")
    }
}