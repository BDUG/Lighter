use crate::prelude::*;


#[derive(Serialize, Deserialize, Debug, Copy, Clone)]
pub enum Optimizers {
    SGD,
    Adam,
    None
}