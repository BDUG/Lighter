#[allow(unused)]
pub use crate::prelude::*;
use candle_transformers::models::bert::{BertModel, Config};

use super::berthiddenacttype::BertHiddenActType;
use super::bertpositionembdtypes::BertPositionEmbeddingType;
use super::transformermodels::TransformerTrait;


pub struct LighterBertModel {
    pub model: BertModel,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: BertHiddenActType,
    pub hidden_dropout_prob: f32,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub initializer_range: f32,
    pub layer_norm_eps: f32,
    pub pad_token_id: usize,
    pub position_embedding_type: BertPositionEmbeddingType,
    pub device: Device,
    pub name: String,
}



impl Fitable for LighterBertModel {
}

impl Predictable for LighterBertModel  {
}

impl ModelSerialization for LighterBertModel  {
}

impl TransformerTrait for LighterBertModel  {
}

impl DoPrediction for LighterBertModel {
    fn predict(&self, x: &Tensor) -> Option<Vec<Tensor>> {
        // Only time not Spatial
        let mut _new_x = x.reshape((x.shape().dims()[0], x.shape().dims()[2] ) ).unwrap();
        _new_x = _new_x.to_dtype(DType::U32).unwrap();
        let token_type_ids = _new_x.zeros_like().unwrap();
        let _rst = self.model.forward(&_new_x, &token_type_ids).unwrap();
        return Some(vec![_rst]);
        //todo!()
    }
}


pub trait LighterBertModelTrait {
    fn new(
        vocab_size:usize, 
        hidden_size: usize, 
        num_hidden_layers: usize, 
        num_attention_heads:usize, 
        intermediate_size: usize, 
        hidden_act: BertHiddenActType,
        hidden_dropout_prob: f32,
        max_position_embeddings: usize,
        type_vocab_size: usize,
        initializer_range: f32,
        layer_norm_eps: f32,
        pad_token_id: usize,
        position_embedding_type: BertPositionEmbeddingType,
        device: &Device,
        varmap : &VarMap,
        name: String
    ) -> Self;
}


impl LighterBertModelTrait for LighterBertModel {
    fn new(vocab_size:usize, 
        hidden_size: usize, 
        num_hidden_layers: usize, 
        num_attention_heads: usize, 
        intermediate_size: usize, 
        hidden_act: BertHiddenActType,
        hidden_dropout_prob: f32,
        max_position_embeddings: usize,
        type_vocab_size: usize,
        initializer_range: f32,
        layer_norm_eps: f32,
        pad_token_id: usize,
        position_embedding_type: BertPositionEmbeddingType,
        device: &Device,
        varmap : &VarMap,
        name: String) -> Self{
            
      
        // Lazy Init
        let vs = VarBuilder::from_varmap(varmap, DType::F32, &device);
        let _hiddentype = match hidden_act {
            BertHiddenActType::Gelu => "gelu",
            BertHiddenActType::GeluApproximate => "geluapproximate",
            BertHiddenActType::Relu => "relu",
        };
   
        let _positionembeddingtype = match position_embedding_type {
            BertPositionEmbeddingType::Absolute => "absolute",
        };

        let mut configstring = String::new();
        configstring.push_str("{");

        configstring.push_str("\"vocab_size\":");
        configstring.push_str(vocab_size.to_string().as_str());
        configstring.push_str(",");

        configstring.push_str("\"hidden_size\":");
        configstring.push_str(hidden_size.to_string().as_str());
        configstring.push_str(",");

        configstring.push_str("\"num_hidden_layers\":");
        configstring.push_str(num_hidden_layers.to_string().as_str());
        configstring.push_str(",");

        configstring.push_str("\"num_attention_heads\":");
        configstring.push_str(num_attention_heads.to_string().as_str());
        configstring.push_str(",");

        configstring.push_str("\"intermediate_size\":");
        configstring.push_str(intermediate_size.to_string().as_str());
        configstring.push_str(",");

        configstring.push_str("\"hidden_act\":");
        configstring.push_str("\"");
        configstring.push_str(_hiddentype.to_string().as_str());
        configstring.push_str("\"");
        configstring.push_str(",");

        configstring.push_str("\"hidden_dropout_prob\":");
        configstring.push_str(hidden_dropout_prob.to_string().as_str());
        configstring.push_str(",");

        configstring.push_str("\"max_position_embeddings\":");
        configstring.push_str(max_position_embeddings.to_string().as_str());
        configstring.push_str(",");

        configstring.push_str("\"type_vocab_size\":");
        configstring.push_str(type_vocab_size.to_string().as_str());
        configstring.push_str(",");

        configstring.push_str("\"initializer_range\":");
        configstring.push_str(initializer_range.to_string().as_str());
        configstring.push_str(",");

        configstring.push_str("\"layer_norm_eps\":");
        configstring.push_str(layer_norm_eps.to_string().as_str());
        configstring.push_str(",");

        configstring.push_str("\"pad_token_id\":");
        configstring.push_str(pad_token_id.to_string().as_str());
        configstring.push_str(",");

        configstring.push_str("\"position_embedding_type\":");
        configstring.push_str("\"");
        configstring.push_str(_positionembeddingtype.to_string().as_str());
        configstring.push_str("\"");

        configstring.push_str("}");

        println!("{}", configstring);
        let config: Config = serde_json::from_str(&configstring).unwrap();

        let _bertmodel = BertModel::load(vs, &config).unwrap();

        Self {
            model: _bertmodel,
            vocab_size: vocab_size,
            hidden_size: hidden_size,
            num_hidden_layers: num_hidden_layers,
            num_attention_heads: num_attention_heads,
            intermediate_size: intermediate_size,
            hidden_act: hidden_act,
            hidden_dropout_prob: hidden_dropout_prob,
            max_position_embeddings: max_position_embeddings,
            type_vocab_size: type_vocab_size,
            initializer_range: initializer_range,
            layer_norm_eps: layer_norm_eps,
            pad_token_id: pad_token_id,
            position_embedding_type: position_embedding_type,
            device : device.clone(),
            name: name.clone(),
        }
    }

}


