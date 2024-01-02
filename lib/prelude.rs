
pub use std::fs::File;
pub use std::io::Write;
pub use std::io::Read;

pub use candle_core::*;
pub use candle_nn::*;
pub use serde::ser::{Serialize, Serializer, SerializeStruct};

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