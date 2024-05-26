
use crate::embeddingtypes::EmbeddingType;
#[allow(unused)]
use crate::prelude::*;
use flatten::embeddinglayer::Embed;
use flatten::embeddinglayer::EmbeddingLayerTrait;
use ndarray_rand::rand_distr::num_traits::ToPrimitive;
use std::collections::HashMap;

/** This example implements initial parts of the steps given on the following site:
 *  https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html 
 */

pub fn simple_s2s() {
    let sentence = "Life is short, eat dessert first";
    let sentence_binding = sentence.replace(",", "");
    let sentence_splitted: Vec<&str> = sentence_binding.split(" ").collect();
    assert_eq!(sentence_splitted, ["Life", "is", "short", "eat", "dessert", "first"]);
    let mut sentence_splitted_sorted = sentence_splitted.clone();

    // ***************************************************
    // #### 1. Embedding an Input Sentence

    // To avoid a simple increasing order do a alphanumerical sorting 
    sentence_splitted_sorted.sort();
    assert_eq!(sentence_splitted_sorted, ["Life", "dessert", "eat", "first", "is", "short"]);
    let mut dc: HashMap<&str, usize> = HashMap::new();
    for (pos, e) in sentence_splitted_sorted.iter().enumerate() {
        dc.insert(e, pos);
    }
    let mut resulttensor : Vec<u32> = Vec::new();
    for (_pos, e) in sentence_splitted.iter().enumerate() {
        let rst = dc.get(e).clone().unwrap();
        resulttensor.push(rst.clone().to_u32().unwrap());
    }
    let varmap = VarMap::new();
    let dev = candle_core::Device::cuda_if_available(0).unwrap();

    let mut name1 = String::new();
    name1.push_str("embed0");
    let inputtensor : Tensor = Tensor::from_vec(resulttensor, (1,6), &dev ).unwrap().clone();
    let embedding_oepration = Embed::new(EmbeddingType::Standard, 6, 16, &dev , &varmap, name1);
    let embedded_sentence : Tensor = embedding_oepration.forward(inputtensor);
    println!("{}",embedded_sentence.to_string());
    let d = embedded_sentence.dims().to_vec()[2];

    // ***************************************************
    // #### 2. Computing the Unnormalized Attention Weights

    let d_q: usize = 24;
    let d_k: usize = 24;
    let d_v: usize = 28;

    let w_query = Tensor::rand(0.0, 1.0, (d_q,d), &dev).unwrap().to_dtype(DType::F32).unwrap();
    let w_key = Tensor::rand(0.0, 1.0, (d_k,d), &dev).unwrap().to_dtype(DType::F32).unwrap();
    let w_value = Tensor::rand(0.0, 1.0, (d_v,d), &dev).unwrap().to_dtype(DType::F32).unwrap();

    let embedded_sentence_vector = embedded_sentence.to_vec3::<f32>().unwrap();

    let mut query_attention_weight = Vec::new();
    for  (_pos, e) in embedded_sentence_vector[0].iter().enumerate(){
        let word: Tensor = Tensor::from_vec(e.clone(), (d,1), &dev ).unwrap().clone();
        query_attention_weight.push(w_query.matmul(&word).unwrap().clone());
    }
    println!("Vector: {:?}", query_attention_weight);

    let keys = w_key.matmul(&embedded_sentence.reshape( (6,16) ).unwrap().t().unwrap()).unwrap();
    let _values = w_value.matmul(&embedded_sentence.reshape( (6,16) ).unwrap().t().unwrap()).unwrap();

    // omega = w
    let mut omegas = Vec::new();
    for  (_pos, e) in query_attention_weight.iter().enumerate(){
        let omega_tmp = e.reshape( (1,24) ).unwrap().matmul(&keys).unwrap();
        omegas.push(omega_tmp.clone());
    }

}

