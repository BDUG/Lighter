
pub use std::io::Write;
pub use std::io::Read;
pub use std::any::Any;

pub use std::fs;
pub use std::fs::*;

pub use candle_core::*;
pub use candle_nn::*;

pub use serde::ser::SerializeStruct;
pub use serde_json::Value;
pub use serde::{Serializer, Serialize, Deserialize};

pub use ndarray::*;
pub use ndarray::prelude::*;
pub use ndarray_rand::RandomExt;
pub use ndarray_rand::rand_distr::Uniform;

pub use crate::layers::*;
pub use crate::models::*;
pub use crate::optimizers::*;
pub use crate::losses::*;
pub use crate::utils::*;
pub use crate::activations::*;

pub use crate::rand_array;
pub use crate::Model;
pub use crate::Dense;


pub trait StringConstruction {
    fn from_string(name: String) -> Self;
}