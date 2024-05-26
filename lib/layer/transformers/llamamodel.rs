
use std::collections::HashMap;

#[allow(unused)]
pub use crate::prelude::*;
use crate::saveweightstype::SaveWeightsType;
use candle_transformers::{generation::{LogitsProcessor, Sampling}, models::llama::{Cache, Llama, LlamaConfig}};


use super::transformermodels::TransformerTrait;


pub struct LighterLLamaModel  {
    pub p: f32,
    pub k: usize,
    pub penaltiy: f32,
    pub temperature: f32,
    pub output_size: usize,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub device: Device,
    pub name: String,
    pub vars: VarMap,
}

impl Fitable for LighterLLamaModel {

}

impl Predictable for LighterLLamaModel {
}

impl ModelSerialization for LighterLLamaModel {
    fn load_weights(&self, weighttype: SaveWeightsType, parameter: &HashMap<String, Value>, varmap: &VarMap, device: &Device) {
        self.default_load_weights(weighttype, parameter, &self.vars, device);
    }
    
}

impl TransformerTrait for LighterLLamaModel  {
}

impl DoPrediction for LighterLLamaModel {


    fn predict(&self, x: &Tensor) -> Option<Vec<Tensor>> {

        // Lazy Init
        let vs = VarBuilder::from_varmap(&self.vars, DType::F32, x.device());
        let config = config(self.vocab_size, self.hidden_size, self.intermediate_size, self.num_hidden_layers, self.num_attention_heads, self.num_key_value_heads, self.rms_norm_eps, self.rope_theta);  
        let model = Llama::load(vs, &config.clone().into_config(false)).unwrap();

        //# use cache
        let mut cache = Cache::new(false, DType::F32, &config.clone().into_config(false), x.device()).unwrap();

        // Only time not Spatial
        let _token_len = x.shape().dims()[2] ;
        let mut _new_x = x.reshape((x.shape().dims()[0], _token_len ) ).unwrap().to_dtype(DType::U32).unwrap();
        let mut tokens: Vec<u32> = _new_x.flatten_all().unwrap().to_dtype(DType::U32).unwrap().to_vec1().unwrap();

        // Logits processor
        let mut _logists_processor = LogitsProcessor::from_sampling(0, Sampling::TopKThenTopP { k: self.k, p: self.p as f64, temperature: self.temperature as f64});
        
        for index in 0..self.output_size {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)
                .unwrap()
                .unsqueeze(0)
                .unwrap();
            let logits = model.forward(&input, start_pos, &mut cache).unwrap();
            let logits = logits
                .squeeze(0)
                .unwrap()
                .squeeze(0)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap();
            let start_at = tokens.len().saturating_sub(self.k);
            let logits = candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.penaltiy,
                    &tokens[start_at..],
                )
                .unwrap();

            let next_token = _logists_processor.sample(&logits).unwrap();
            tokens.push(next_token);

            if Some(next_token) == None {
                break;
            }
        }

        let _rst = Tensor::new(tokens.clone(), x.device()).unwrap();
        return Some(vec![_rst]);
    }

}


pub trait LighterLLamaModelTrait {
    fn new(
        p: f32,
        k: usize,
        penaltiy: f32,
        temperature: f32,
        output_size: usize,
        vocab_size:usize, 
        hidden_size: usize, 
        intermediate_size: usize, 
        num_hidden_layers: usize, 
        num_attention_heads: usize, 
        num_key_value_heads: usize, 
        rms_norm_eps: f32,
        rope_theta: f32,
        device: &Device,
        name: String,
    ) -> Self;
}



impl LighterLLamaModelTrait for LighterLLamaModel {
    fn new(
        p: f32,
        k: usize,
        penaltiy: f32,
        temperature: f32,
        output_size: usize,
        vocab_size: usize, 
        hidden_size: usize, 
        intermediate_size: usize, 
        num_hidden_layers: usize, 
        num_attention_heads: usize, 
        num_key_value_heads: usize, 
        rms_norm_eps: f32,
        rope_theta: f32,
        device: &Device,
        name: String) -> Self{
            
      
        Self {
            p: p,
            k: k,
            penaltiy: penaltiy,
            temperature: temperature,
            output_size: output_size,
            vocab_size: vocab_size,
            hidden_size: hidden_size,
            intermediate_size: intermediate_size,
            num_hidden_layers: num_hidden_layers,
            num_attention_heads: num_attention_heads,
            num_key_value_heads: num_key_value_heads,
            rms_norm_eps: rms_norm_eps,
            rope_theta: rope_theta,
            device : device.clone(),
            name: name.clone(),
            vars: VarMap::new(),
        }
    }

}



fn config(vocab_size: usize, hidden_size: usize, intermediate_size: usize, num_hidden_layers: usize, num_attention_heads: usize, num_key_value_heads: usize, rms_norm_eps: f32, rope_theta: f32) -> LlamaConfig {
    let mut configstring = String::new();
    configstring.push_str("{");

    configstring.push_str("\"vocab_size\":");
    configstring.push_str(vocab_size.to_string().as_str());
    configstring.push_str(",");

    configstring.push_str("\"hidden_size\":");
    configstring.push_str(hidden_size.to_string().as_str());
    configstring.push_str(",");

    configstring.push_str("\"intermediate_size\":");
    configstring.push_str(intermediate_size.to_string().as_str());
    configstring.push_str(",");

    configstring.push_str("\"num_hidden_layers\":");
    configstring.push_str(num_hidden_layers.to_string().as_str());
    configstring.push_str(",");

    configstring.push_str("\"num_attention_heads\":");
    configstring.push_str(num_attention_heads.to_string().as_str());
    configstring.push_str(",");

    configstring.push_str("\"num_key_value_heads\":");
    configstring.push_str(num_key_value_heads.to_string().as_str());
    configstring.push_str(",");

    configstring.push_str("\"rms_norm_eps\":");
    configstring.push_str(rms_norm_eps.to_string().as_str());
    configstring.push_str(",");

    configstring.push_str("\"rope_theta\":");
    configstring.push_str(rope_theta.to_string().as_str());

    configstring.push_str("}");
        
    let config: LlamaConfig = serde_json::from_str(&configstring).unwrap();
    config
}


