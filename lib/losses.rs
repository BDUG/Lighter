
use crate::prelude::*;

#[derive(Serialize, Deserialize, Debug, Copy, Clone)]
pub enum Loss {
    MSE,
    NLL,
    BinaryCrossEntropyWithLogit,
    CrossEntropy,
    None,
}
