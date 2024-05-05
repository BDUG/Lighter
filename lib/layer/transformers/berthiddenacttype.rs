#[allow(unused)]
use crate::prelude::*;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BertHiddenActType {
    Gelu,
    GeluApproximate,
    Relu,
}


impl StringConstruction for BertHiddenActType {
    fn from_string(name: String) -> Self {
        if name.to_lowercase().eq("gelu"){
            return BertHiddenActType::Gelu;
        }
        else if name.to_lowercase().eq("geluapproximate"){
            return BertHiddenActType::GeluApproximate;
        }
        else if name.to_lowercase().eq("relu"){
            return BertHiddenActType::Relu;
        }
        panic!("Unknown Bert hidden act type type")
    }
}