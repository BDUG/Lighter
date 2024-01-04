

#[allow(unused)]
use crate::prelude::*;

#[derive(Serialize, Deserialize)]
pub struct SerializedTensor {
    pub name: String,
    pub dimension : Vec<usize>,
    pub values : Vec<f32>,
}