
#[allow(unused)]
use crate::prelude::*;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParallelModelType {
    Split,
    Merge
}


impl StringConstruction for ParallelModelType {
    fn from_string(name: String) -> Self {
        if name.to_lowercase().eq("split"){
            return ParallelModelType::Split;
        }
        else if name.to_lowercase().eq("merge"){
            return ParallelModelType::Merge;
        }
        panic!("Unknown optimizer {}",name)
    }
}