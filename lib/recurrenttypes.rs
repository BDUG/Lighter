#[allow(unused)]
use crate::prelude::*;


#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecurrentType {
    LSTM,
    GRU,
}


impl StringConstruction for RecurrentType {
    fn from_string(name: String) -> Self {
        if name.to_lowercase().eq("lstm"){
            return RecurrentType::LSTM;
        }
        else if name.to_lowercase().eq("gru"){
            return RecurrentType::GRU;
        }
        panic!("Unknown recurrent type")
    }
}