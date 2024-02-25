use std::ops::Add;

use crate::embeddingtypes::EmbeddingType;
#[allow(unused)]
use crate::prelude::*;
use crate::recurrenttypes::RecurrentType;
use flatten::embeddinglayer::Embed;
use flatten::embeddinglayer::EmbeddingLayerTrait;
use ndarray::prelude::*;
use ndarray::Array;
use ndarray_rand::rand_distr::num_traits::ToPrimitive;
use rand::Rng;
use std::collections::HashMap;

/**
This examples bases on the following site: https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html 
*/

pub fn simple_s2s() {
    let sentence = "Life is short, eat dessert first";
    let sentence_binding = sentence.replace(",", "");
    let sentence_splitted: Vec<&str> = sentence_binding.split(" ").collect();
    assert_eq!(sentence_splitted, ["Life", "is", "short", "eat", "dessert", "first"]);
    let mut sentence_splitted_sorted = sentence_splitted.clone();
    // To avoid a simple increasing order do a alphanumerical sorting 
    sentence_splitted_sorted.sort();
    assert_eq!(sentence_splitted_sorted, ["Life", "dessert", "eat", "first", "is", "short"]);
    let mut dc: HashMap<&str, usize> = HashMap::new();
    for (pos, e) in sentence_splitted_sorted.iter().enumerate() {
        dc.insert(e.clone(), pos);
    }
    let mut resulttensor : Vec<u32> = Vec::new();
    for (pos, e) in sentence_splitted.iter().enumerate() {
        let rst = dc.get(e).clone().unwrap();
        resulttensor.push(rst.clone().to_u32().unwrap());
    }
    let varmap = VarMap::new();
    let dev = candle_core::Device::cuda_if_available(0).unwrap();

    let mut name1 = String::new();
    name1.push_str("embed0");

    let mut layers: Vec<Box<dyn Trainable>> = vec![];

    let inputtensor : Tensor = Tensor::from_vec(resulttensor, (1,6), &dev ).unwrap().clone();

    let embedding_oepration = Embed::new(EmbeddingType::Standard, 6, 16, &dev , &varmap, name1);
    let outputtensor : Tensor = embedding_oepration.forward(inputtensor);

    println!("{}", outputtensor.to_string());
}

