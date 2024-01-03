use serde::{Serialize, Deserialize};

#[allow(unused)]
use crate::prelude::*;


#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Optimizers {
    SGD(f64),
    Adam(f64),
    None(f64)
}


impl StringConstruction for Optimizers {
    fn from_string(name: String) -> Self {
        if name.to_lowercase().eq("sgd"){
            return Optimizers::SGD(0.01);
        }
        else if name.to_lowercase().eq("adam"){
            return Optimizers::Adam(0.01);
        }
        else if name.to_lowercase().eq("none"){
            return Optimizers::None(0.01);
        }
        panic!("Unknown optimizer")
    }
}