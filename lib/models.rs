

use std::cell::Cell;
use std::{fs, borrow::Borrow};
use serde::{Serializer};
use serde_json::Value;
extern crate lazy_static;
use crate::prelude::*;
use std::any::{Any, TypeId};
use std::any::type_name;

pub struct Sequential {
    pub layers: Vec<Dense> ,
    pub optimizer: Optimizers,
    pub loss: Loss,
    pub varmap: VarMap,
}

impl Sequential {
    pub fn new(varmap: VarMap,layers: Vec<Dense>) -> Self {

        Self {
            layers: layers,
            optimizer: Optimizers::None,
            loss: Loss::None,
            varmap: varmap,
        }
    }

    pub fn summary(&self) {
        // TODO
    }

    pub fn compile(&mut self, optimizer: Optimizers, loss: Loss) {
        self.optimizer = optimizer;
        self.loss = loss;
    }

    pub fn fit(&mut self, x: Tensor, _y: Tensor, epochs: usize ,_verbose: bool) {
        for _i in 0..epochs {
            if _verbose{
                println!("Epoche {}",_i);
            }
            
            for elementnumber in 0.. x.dims().len() {
                let mut input_checked = match x.get(elementnumber) {
                    Ok(element) => element,
                    Err(error) => panic!("{}",error.to_string()),
                };

                let output_checked = match _y.get(elementnumber) {
                    Ok(element) => element,
                    Err(error) => panic!("{}",error.to_string()),
                };
                
                //let mut input_stack = vec![];
                for layer in self.layers.iter() {
                    input_checked = layer.forward(input_checked).clone();
                }
                // TODO: Apply loss
                let lossed =  match self.loss {
                    Loss::MSE => candle_nn::loss::mse(&input_checked, &output_checked),
                    Loss::NLL => candle_nn::loss::nll(&input_checked, &output_checked),
                    Loss::BinaryCrossEntropyWithLogit => candle_nn::loss::binary_cross_entropy_with_logit(&input_checked, &output_checked),
                    Loss::CrossEntropy => candle_nn::loss::cross_entropy(&input_checked, &output_checked),
                    Loss::None => todo!(),
                };

                let lossed_checked = match lossed {
                    Ok(lossed) => lossed,
                    Err(error) => panic!("{}",error.to_string()),
                };

                let enumvalue: u8 = match self.optimizer {
                    Optimizers::SGD =>  1,
                    Optimizers::Adam => 2,
                    Optimizers::None => 0,
                };
                // Apply optimizer 
                // Also see https://github.com/huggingface/candle/issues/1509#issuecomment-1872916766
                if enumvalue == 1 {
                    let mut optimized: SGD = candle_nn::SGD::new(self.varmap.all_vars(), 0.01).unwrap();
                    optimized.backward_step(&lossed_checked);
                }
                else if enumvalue == 2 {
                    let adamw_params = candle_nn::ParamsAdamW {
                        lr: 0.0001,
                        ..Default::default()
                    };
                    let mut optimized: AdamW = candle_nn::AdamW::new(self.varmap.all_vars(), adamw_params).unwrap();
                    optimized.backward_step(&lossed_checked);
                }
            }
        }
    }


    pub fn predict(&self, mut x: Tensor) -> Tensor {
        for layer in self.layers.iter() {
            x = layer.forward(x);
        }
        return x;
    }

    pub fn save_model(&self, path: &str) {
        let mut file = File::create(path).unwrap();
        let j = serde_json::to_string(&self).unwrap();
        file.write(&j.as_bytes()).unwrap();
    }


    fn extractjson_value_u64(&self, elem: &serde_json::Map<String, Value>, key: &str) -> u64{
        let value: Option<(&String, &Value)> = elem.get_key_value(key);
        let rst = match value {
            Some(x) => x.1.clone(),
            None    => panic!("Unknown state"),
        };
        if rst.is_u64() {
            return u64::from(rst.as_u64().unwrap());
        }
        panic!("Unknown type");
    }

    fn extractjson_value_str(&self, elem: &serde_json::Map<String, Value>, key: &str)-> String {
        let value: Option<(&String, &Value)> = elem.get_key_value(key);
        let rst = match value {
            Some(x) => x.1.clone(),
            None    => panic!("Unknown state"),
        };
        if rst.is_string(){
            return String::from(rst.as_str().unwrap());
        }
        panic!("Unknown type");
    }

    pub fn load_model(&self, path: &str, device : &Device) -> Sequential {
        let value = fs::read_to_string(path).unwrap();
        let json: serde_json::Value =
            serde_json::from_str(&value).unwrap();
        
        let _value_map = match json {
            Value::Object(ref map) => map,
            _ => panic!("Unknown state"),
        };
        let numberoflayers_string = _value_map.get_key_value("layers");
        let rst = match numberoflayers_string {
            Some(x) => x.1,
            None    => panic!("Unknown state"),
        };
        let varmap = VarMap::new();
        let mut layers = vec![];
        let layerslist = rst.as_array().unwrap();
        for i in 0..layerslist.len(){
            let elem = layerslist.get(i).unwrap().as_object().unwrap();
            let new_perceptrons = self.extractjson_value_u64(elem, "perceptrons");
            let new_previousperceptrons = self.extractjson_value_u64(elem, "previousperceptrons");
            let new_name = self.extractjson_value_str(elem, "name");
            let tmp_activation = self.extractjson_value_str(elem, "activation");
            let new_activation = Activations::from_string(tmp_activation.to_string());
            
            layers.push(Dense::new(new_perceptrons as usize, new_previousperceptrons as usize, new_activation, device, &self.varmap, new_name));
        }

        let new_optimizer = self.extractjson_value_str(_value_map, "optimizer");
        let new_loss = self.extractjson_value_str(_value_map, "loss");

        let mut new_model= Sequential::new(varmap,layers);
        new_model.compile(Optimizers::from_string(new_optimizer), Loss::from_string(new_loss));
        return new_model;
    }
  
    pub fn buildname(&self, name: &str,  number: i32) -> String {
        let mut result = String::from(name);
        result.push_str(&number.to_string());
        return result;
    }
  

}

impl Serialize for Sequential {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<<S as Serializer>::Ok, <S as Serializer>::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("sequential", 3)?;
        state.serialize_field("optimizer", &self.optimizer);
        state.serialize_field("loss", &self.loss);      
        state.serialize_field("layers", &self.layers);
        return Ok(state.end().unwrap());
    }
}
