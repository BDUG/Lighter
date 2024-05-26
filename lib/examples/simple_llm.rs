
use std::{collections::HashMap};

#[allow(unused)]
use crate::prelude::*;
use crate::{preprocessing::features::{Features, FeaturesTrait}, saveweightstype::SaveWeightsType};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use selfattention::flatten::transformers::{tokeoutputstream::TokenOutputStream, transformermodels::TransformerTrait};
use tokenizers::PaddingParams;

use self::selfattention::flatten::transformers::{berthiddenacttype::BertHiddenActType, bertmodel::{LighterBertModel, LighterBertModelTrait}, bertpositionembdtypes::BertPositionEmbeddingType, llamamodel::{LighterLLamaModel, LighterLLamaModelTrait}};


pub fn to_tensor(input: &Vec<Vec<f32>>, device: &Device) -> Tensor{
    let dimension1: usize = input.len();
    let dimension2: usize = input.get(0).unwrap().len();
    let mut result = Vec::new();
    for i in 0..dimension1 {
        for j in 0..dimension2 {
            let val = input.get(i).unwrap().get(j).unwrap();
            result.push(val.clone().to_owned());
        }
    }
    return Tensor::from_vec(result, (dimension1,1,dimension2), device ).unwrap().clone();
}
pub fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}

pub fn simple_llm() {

    let varmap = VarMap::new();
    let dev = candle_core::Device::cuda_if_available(0).unwrap();
    
    // Adapted from https://github.com/huggingface/candle/blob/a0facd0e67b546215ea62b53dc28a1cb2e6dcd47/candle-examples/examples/bert/main.rs
    let sentences = [
            "The cat sits outside",
            "A man is playing guitar",
            "I love pasta",
            "The new movie is awesome",
            "The cat plays in the garden",
            "A woman watches TV",
            "The new movie is so great",
            "Do you like pizza?",
     ];
    let pp = PaddingParams {
        strategy: tokenizers::PaddingStrategy::BatchLongest,
        ..Default::default()
    };

    let _bert_model = LighterBertModel::new(
        30522, 
        768, 
        12, 
        12, 
        3072, 
        BertHiddenActType::Gelu, 
        0.1, 
        512, 
        2, 
        0.02, 
        1e-12, 
        0, 
        BertPositionEmbeddingType::Absolute, 
        &dev, 
        &varmap, 
        "bert1".to_string());

    let mut parameter = HashMap::new();
    parameter.insert("hf_token".into(), Value::String("<YOUR TOKEN>".into()));
    parameter.insert("hf_model".into(), Value::String("sentence-transformers/all-MiniLM-L6-v2".into()));
    // Or your pathes after first download
    // parameter.insert("hf_model.safetensors".into(), Value::String("<YOUR PATH>".into()));
    // parameter.insert("hf_tokenizer.json".into(), Value::String("<YOUR PATH>".into()));

    

    let mut _tokenizer =  _bert_model.get_tokenizer(&parameter);    

    _tokenizer.with_padding(Some(pp));
    let tokens = _tokenizer
        .encode_batch(sentences.to_vec(), true).unwrap();
    let token_ids = tokens
        .iter()
        .map(|tokens| {
            let tokens = tokens.get_ids().to_vec();
            Ok(Tensor::new(tokens.as_slice(), &dev)?)
        })
        .collect::<Result<Vec<_>>>().unwrap();

    _bert_model.load_weights(SaveWeightsType::HuggingfaceHub, &parameter, &varmap, &dev); 

    let mut featurehelper_x_test = Features::new(dev.clone());
    for element in token_ids{
        featurehelper_x_test.add_feature(element);
    }
   
    let input = &featurehelper_x_test.get_data_tensor();
    let result = _bert_model.predict(input).unwrap();
    let embeddings = result.get(0).unwrap();
    let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3().unwrap();
    let n_sentences = sentences.len();
    let embeddings = (embeddings.sum(1).unwrap() / (n_tokens as f64)).unwrap();
   
    let mut similarities = vec![];
    for i in 0..n_sentences {
        let e_i = embeddings.get(i).unwrap();
        for j in (i + 1)..n_sentences {
            let e_j = embeddings.get(j).unwrap();
            let sum_ij = (&e_i * &e_j).unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
            let sum_i2 = (&e_i * &e_i).unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
            let sum_j2 = (&e_j * &e_j).unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
            let cosine_similarity = sum_ij / (sum_i2 * sum_j2).sqrt();
            similarities.push((cosine_similarity, i, j))
        }
    }
    similarities.sort_by(|u, v| v.0.total_cmp(&u.0));
    for &(score, i, j) in similarities[..5].iter() {
        println!("score: {score:.2} '{}' '{}'", sentences[i], sentences[j])
    }

}


// https://huggingface.co/tasks
// one task is https://huggingface.co/tasks/text-generation
pub fn simple_llm2() {

    let varmap = VarMap::new();
    let dev = candle_core::Device::cuda_if_available(0).unwrap();
    
    // TinyLlama-1.1B-Chat-v1.0 
    let llamamodel = LighterLLamaModel::new(
        0.85,
        20,
        3.2,
        0.7,
        50,
        32000, 
        768, 
        3072, 
        12, 
        12, 
        12, 
        1e-06, 
        1.0, 
        &dev, 
        "llama1".to_string());

    let mut parameter = HashMap::new();
    parameter.insert("hf_token".into(), Value::String("<YOUR TOKEN>".into()));
    parameter.insert("hf_model".into(), Value::String("JackFram/llama-160m".into()));
    // Or your pathes after first download
    // parameter.insert("hf_model.safetensors".into(), Value::String("<YOUR PATH>".into()));
    // parameter.insert("hf_tokenizer.json".into(), Value::String("<YOUR PATH>".into()));

    let mut _tokenizer =  llamamodel.get_tokenizer(&parameter);    
    llamamodel.load_weights(SaveWeightsType::HuggingfaceHub, &parameter, &varmap, &dev); 

    let pp = PaddingParams {
        strategy: tokenizers::PaddingStrategy::BatchLongest,
        ..Default::default()
    };
    _tokenizer.with_padding(Some(pp));

    let tokens = _tokenizer.encode("The world is ", true).unwrap();
    let mut token_ids = tokens.get_ids().to_vec();
    println!("Given query: {}",_tokenizer.decode(&token_ids, true).unwrap());

    let mut _tokenizer_stream = TokenOutputStream::new(_tokenizer.clone());

    let mut featurehelper_x_test = Features::new(dev.clone());
    let _tmp: Vec<f32> = token_ids.iter().map(|&e| e as f32).collect();
    let _tmp2= Tensor::new(_tmp, &dev).unwrap();
    featurehelper_x_test.add_feature(_tmp2);


    let input = &featurehelper_x_test.get_data_tensor();
    let result = llamamodel.predict(input).unwrap();

    let vv= result.get(0).unwrap().flatten_all().unwrap().to_vec1().unwrap();


    let mut string = String::new();
    for _token in vv {
        if let t = _tokenizer_stream.next_token(_token).unwrap() {
            if t.clone() != None {
                string.push_str(&t.unwrap());
            }
        }
    }
    println!("Given answer: {}", string);
   
}

