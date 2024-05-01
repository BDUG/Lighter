use std::fmt::Debug;

use rand::distributions::Distribution;

#[allow(unused)]
use crate::prelude::*;
use crate::preprocessing::{self, features::{Features, FeaturesTrait}};

pub struct Dataitem {
    x: Vec<usize>,
    y: Vec<usize>
}

pub fn generatedata(numofelements: usize, limit: usize) -> Vec<Dataitem> {
    let mut result: Vec<Dataitem> = vec![];

    let vals: Vec<u64> = (0..numofelements as u64).collect();
    for (_i,_valuee) in vals.iter().enumerate() {
        let index = Uniform::new(0, limit);
        let mut rng = rand::thread_rng();
        let a = index.sample(&mut rng);
        let b = index.sample(&mut rng);

        let mut resultelement =  Dataitem {
            x: Vec::new(), // e.g., 1. , 2.
            y: Vec::new() // e.g., 3.
        };
        resultelement.x.push(a);
        resultelement.x.push(b);

        resultelement.y.push( a+b );

        result.push(resultelement);
    }
    return result;
}


pub fn simple_dnn() {
    let varmap = VarMap::new();
    let dev = candle_core::Device::cuda_if_available(0).unwrap();

    let dataset = generatedata(2000, 10);
    let mut featurehelper_x = Features::new(dev.clone());
    let mut featurehelper_y = Features::new(dev.clone());

    for (_j, value) in dataset.iter().enumerate() {
        let tmp_x_value = value.x.iter().filter_map( |s| Some(*s as f32) ) .collect();
        featurehelper_x.add_feature_1_d(tmp_x_value);

        let tmp_y_value = value.y.iter().filter_map( |s| Some(*s as f32) ) .collect();
        featurehelper_y.add_feature_1_d(tmp_y_value);
    }
    let tmp_x = featurehelper_x.get_data_tensor();
    let tmp_y = featurehelper_y.get_data_tensor();

    //println!("{}",tmp_x);
    //println!("{}",tmp_y);

    let mut layers: Vec<Box<dyn Trainable>> = vec![];
    let mut name1 = String::new();
    name1.push_str("fc1");
    layers.push(Box::new(Dense::new(20, 2, Activations::Relu, &dev, &varmap, name1 )));
    let mut name2 = String::new();
    name2.push_str("fc2");
    layers.push(Box::new(Dense::new(2, 20, Activations::Relu, &dev, &varmap, name2 )));
    let mut name3 = String::new();
    name3.push_str("fc3");
    layers.push(Box::new(Dense::new(1, 2, Activations::Relu, &dev, &varmap, name3 )));

    let mut model = SequentialModel::new(varmap, layers);
    
    let numbers: Vec<f32> = (0..=100).map(|x| x as f32).collect();
    let scaling = preprocessing::featurescaling::FeatureScaling::new(Tensor::new( numbers, &dev).unwrap());

    model.compile(Optimizers::SGD(0.0001), Loss::MSE);
    model.fit(
        scaling.min_max_normalization_other(tmp_x), 
        scaling.min_max_normalization_other(tmp_y), 
        1000, 
        false);
    
    let mut featurehelper_x_test = Features::new(dev.clone());
    //let x_test: [[f32; 2]; 1] = [ [4., 5.] ];
    let x_test: [f32; 2] = [4., 5.];
    let _tmp_tensor = Tensor::new(&x_test, &dev).unwrap();
    featurehelper_x_test.add_feature(_tmp_tensor);

    let tmp_tensor = scaling.min_max_normalization_other(featurehelper_x_test.get_data_tensor());
    let prediction = model.predict(tmp_tensor);
    println!("Prediction: {}", scaling.min_max_normalization_reverse( prediction.get(0).unwrap().clone() ));
}



pub fn simple_dnn2() {
    let varmap = VarMap::new();
    let dev = candle_core::Device::cuda_if_available(0).unwrap();

    let dataset = generatedata(2000, 10);
    let mut featurehelper_x = Features::new(dev.clone());
    let mut featurehelper_y = Features::new(dev.clone());

    for (_j, value) in dataset.iter().enumerate() {
        let tmp_x_value = value.x.iter().filter_map( |s| Some(*s as f32) ) .collect();
        featurehelper_x.add_feature_1_d(tmp_x_value);

        let tmp_y_value = value.y.iter().filter_map( |s| Some(*s as f32) ) .collect();
        featurehelper_y.add_feature_1_d(tmp_y_value);
    }
    let tmp_x = featurehelper_x.get_data_tensor();
    let tmp_y = featurehelper_y.get_data_tensor();

    //println!("{}",tmp_x);
    //println!("{}",tmp_y);
    //println!("{}",tmp_x.get(0).unwrap());

    let mut _rank = Tensor::rand(0.0, 1.0, tmp_x.get(0).unwrap().shape(), &dev).unwrap().to_dtype(DType::F32).unwrap();
    let mut _alpha = 0.25;

    let mut layers: Vec<Box<dyn Trainable>> = vec![];
    let mut name1 = String::new();
    name1.push_str("fc1");
    layers.push(Box::new(Dense::new2(20, 2, Activations::Relu, crate::densetypes::DenseType::LORA,_rank.clone(), _alpha, &dev, &varmap, name1 )));
    let mut name2 = String::new();
    name2.push_str("fc2");
    layers.push(Box::new(Dense::new(2, 20, Activations::Relu,  &dev, &varmap, name2 )));
    let mut name3 = String::new();
    name3.push_str("fc3");
    layers.push(Box::new(Dense::new(1, 2, Activations::Relu,&dev, &varmap, name3 )));

    let mut model = SequentialModel::new(varmap, layers);
    
    let numbers: Vec<f32> = (0..=100).map(|x| x as f32).collect();
    let scaling = preprocessing::featurescaling::FeatureScaling::new(Tensor::new( numbers, &dev).unwrap());

    model.compile(Optimizers::SGD(0.0001), Loss::MSE);
    model.fit(
        scaling.min_max_normalization_other(tmp_x), 
        scaling.min_max_normalization_other(tmp_y), 
        10, 
        false);
    
    let mut featurehelper_x_test = Features::new(dev.clone());
    //let x_test: [[f32; 2]; 1] = [ [4., 5.] ];
    let x_test: [f32; 2] = [4., 5.];
    let _tmp_tensor = Tensor::new(&x_test, &dev).unwrap();
    featurehelper_x_test.add_feature(_tmp_tensor);

    let tmp_tensor = scaling.min_max_normalization_other(featurehelper_x_test.get_data_tensor());
    let prediction = model.predict(tmp_tensor);
    println!("Prediction: {}", scaling.min_max_normalization_reverse( prediction.get(0).unwrap().clone() ));
}



pub fn simple_dnn3() {
    let varmap = VarMap::new();
    let dev = candle_core::Device::cuda_if_available(0).unwrap();

    let dataset = generatedata(2000, 10);
    let mut featurehelper_x = Features::new(dev.clone());
    let mut featurehelper_y = Features::new(dev.clone());

    for (_j, value) in dataset.iter().enumerate() {
        let tmp_x_value = value.x.iter().filter_map( |s| Some(*s as f32) ) .collect();
        featurehelper_x.add_feature_1_d(tmp_x_value);

        let tmp_y_value = value.y.iter().filter_map( |s| Some(*s as f32) ) .collect();
        featurehelper_y.add_feature_1_d(tmp_y_value);
    }
    let tmp_x = featurehelper_x.get_data_tensor();
    let tmp_y = featurehelper_y.get_data_tensor();

    //println!("{}",tmp_x);
    //println!("{}",tmp_y);
    //println!("{}",tmp_x.get(0).unwrap());

    let mut _rank = Tensor::rand(0.0, 1.0, tmp_x.get(0).unwrap().shape(), &dev).unwrap().to_dtype(DType::F32).unwrap();
    let mut _alpha = 0.25;

    let mut layers: Vec<Box<dyn Trainable>> = vec![];
    let mut name1 = String::new();
    name1.push_str("fc1");
    layers.push(Box::new(Dense::new2(20, 2, Activations::Relu, crate::densetypes::DenseType::DORA,_rank.clone(), _alpha, &dev, &varmap, name1 )));
    let mut name2 = String::new();
    name2.push_str("fc2");
    layers.push(Box::new(Dense::new(2, 20, Activations::Relu,  &dev, &varmap, name2 )));
    let mut name3 = String::new();
    name3.push_str("fc3");
    layers.push(Box::new(Dense::new(1, 2, Activations::Relu,&dev, &varmap, name3 )));

    let mut model = SequentialModel::new(varmap, layers);
    
    let numbers: Vec<f32> = (0..=100).map(|x| x as f32).collect();
    let scaling = preprocessing::featurescaling::FeatureScaling::new(Tensor::new( numbers, &dev).unwrap());

    model.compile(Optimizers::SGD(0.0001), Loss::MSE);
    model.fit(
        scaling.min_max_normalization_other(tmp_x), 
        scaling.min_max_normalization_other(tmp_y), 
        100, 
        false);
    
    let mut featurehelper_x_test = Features::new(dev.clone());
    //let x_test: [[f32; 2]; 1] = [ [4., 5.] ];
    let x_test: [f32; 2] = [4., 5.];
    let _tmp_tensor = Tensor::new(&x_test, &dev).unwrap();
    featurehelper_x_test.add_feature(_tmp_tensor);

    let tmp_tensor = scaling.min_max_normalization_other(featurehelper_x_test.get_data_tensor());
    let prediction = model.predict(tmp_tensor);
    println!("Prediction: {}", scaling.min_max_normalization_reverse( prediction.get(0).unwrap().clone() ));
}
