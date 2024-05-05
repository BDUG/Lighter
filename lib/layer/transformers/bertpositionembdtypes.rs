#[allow(unused)]
use crate::prelude::*;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BertPositionEmbeddingType {
    Absolute,
}


impl StringConstruction for BertPositionEmbeddingType {
    fn from_string(name: String) -> Self {
        if name.to_lowercase().eq("absolute"){
            return BertPositionEmbeddingType::Absolute;
        }
        panic!("Unknown positional encoding type")
    }
}