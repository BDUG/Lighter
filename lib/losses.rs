use serde::{Serialize, Deserialize};

#[allow(unused)]
use crate::prelude::*;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Loss {
    MSE,
    NLL,
    BinaryCrossEntropyWithLogit,
    CrossEntropy,
    None,
}

impl StringConstruction for Loss {
    fn from_string(name: String) -> Self {
        if name.to_lowercase().eq("mse"){
            return Loss::MSE;
        }
        else if name.to_lowercase().eq("nll"){
            return Loss::NLL;
        }
        else if name.to_lowercase().eq("binarycrossentropywithlLogit"){
            return Loss::BinaryCrossEntropyWithLogit;
        }
        else if name.to_lowercase().eq("crossentropy"){
            return Loss::CrossEntropy;
        }
        else if name.to_lowercase().eq("none"){
            return Loss::None;
        }
        panic!("Unknown loss")
    }
}