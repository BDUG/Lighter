use serde::{Serialize, Deserialize};

#[allow(unused)]
use crate::prelude::*;


#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Optimizers {
    SGD(f64),
    Adam,
    None
}


impl StringConstruction for Optimizers {
    fn from_string(name: String) -> Self {
        if name.to_lowercase().eq("sgd"){
            return Optimizers::SGD(0.01);
        }
        else if name.to_lowercase().eq("adam"){
            return Optimizers::Adam;
        }
        else if name.to_lowercase().eq("none"){
            return Optimizers::None;
        }
        panic!("Unknown optimizer")
    }
}