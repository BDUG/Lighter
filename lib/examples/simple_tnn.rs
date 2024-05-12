
#[allow(unused)]
use crate::prelude::*;
use crate::{preprocessing::features::{Features, FeaturesTrait}, recurrenttypes::RecurrentType};
use ndarray_rand::rand_distr::num_traits::ToPrimitive;
use rand::distributions::Distribution;
use crate::preprocessing;

/** This example base on the idea of the following site: https://github.com/javierlorenzod/pytorch-attention-mechanism
 * 
 * A sequence of numbers has given delimiter e.g., 0. The numbers after the delimiter will be added e.g.,
 * 
 * 1 2 3 4 0 5 0 7. 
 * 
 * 5+7 = 12 
 */

pub struct TNNDataitem {
    x: Vec<usize>,
    y: usize
}

pub fn generatedata(sizeofsequence: usize, numofelements: usize, delimiter: f32) -> Vec<TNNDataitem> {
    let mut result: Vec<TNNDataitem> = vec![];

    let vals: Vec<u64> = (0..numofelements as u64).collect();
    for (_i, _value) in vals.iter().enumerate() {
        let index_1 = Uniform::new(0, sizeofsequence/ 2);
        let index_2 = Uniform::new(sizeofsequence/2, sizeofsequence);
   
        let mut rng = rand::thread_rng();
        let a = index_1.sample(&mut rng);
        let b = index_2.sample(&mut rng);


        let mut resultelement =  TNNDataitem {
            x: Vec::new(),
            y : a+b
        };

        let vals2: Vec<u64> = (0..sizeofsequence as u64).collect();
        for (_j, value2) in vals2.iter().enumerate() {

            if value2.to_usize() == Some(a) || value2.to_usize() == Some(b) {
                resultelement.x.push(delimiter as usize);
            }
            else{
                resultelement.x.push(value2.to_usize().unwrap());
            }
        }
        
        result.push(resultelement);
    }
    return result;
}


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

pub fn simple_tnn() {
    let sizeofsequence = 20;
    let numofelements = 20000;

    let varmap = VarMap::new();
    let dev = candle_core::Device::cuda_if_available(0).unwrap();
    let dataset = generatedata(sizeofsequence, numofelements, 0.0);

    let mut featurehelper_x = Features::new(dev.clone());
    let mut featurehelper_y = Features::new(dev.clone());

    for (_j, value) in dataset.iter().enumerate() {
        let tmp_x_value = value.x.iter().filter_map( |s| s.to_f32() ) .collect();
        featurehelper_x.add_feature_1_d(tmp_x_value);

        let mut tmp_y_value: Vec<f32> = Vec::new();
        tmp_y_value.push(value.y.to_f32().unwrap());
        featurehelper_y.add_feature_1_d(tmp_y_value);
    }


    let mut layers: Vec<Box<dyn Trainable>> = vec![];

    let mut name1 = String::new();
    name1.push_str("rnn1");
    layers.push(Box::new(Recurrent::new(RecurrentType::LSTM,sizeofsequence, sizeofsequence, &dev, &varmap, name1 )));

    let mut name2 = String::new();
    name2.push_str("attention1");
    // Query dim tells us what the attention sees from the given sequence 
    layers.push(Box::new(SelfAttention::new( 1, 1, sizeofsequence, sizeofsequence, &dev, &varmap,  name2)));
    
    let mut name3 = String::new();
    name3.push_str("fc1");
    layers.push(Box::new(Dense::new(4, sizeofsequence, Activations::Relu, &dev, &varmap, name3 )));
    
    let mut name4 = String::new();
    name4.push_str("fc2");
    layers.push(Box::new(Dense::new(1, 4, Activations::Relu, &dev, &varmap, name4 )));

    let mut model = SequentialModel::new(varmap, layers);
    model.compile(Optimizers::Adam(0.0005), Loss::MSE);       

    let numbers: Vec<f32> = (0..=100).map(|x| x as f32).collect();
    let scaling = preprocessing::featurescaling::FeatureScaling::new(Tensor::new( numbers, &dev).unwrap());

    let tmp_x = featurehelper_x.get_data_tensor();
    let tmp_y = featurehelper_y.get_data_tensor();

    model.fit( 
        scaling.min_max_normalization_other(tmp_x), 
        scaling.min_max_normalization_other(tmp_y), 
        10, 
        false);
    

    
    let mut featurehelper_x_test = Features::new(dev.clone());
    let x_test: [f32; 20] = [1., 2., 3., 4., 0., 6., 7., 8., 9., 0. ,11., 12.,12., 13., 14., 25., 16., 17., 18., 19., ];
    let _tmp_tensor = Tensor::new(&x_test, &dev).unwrap();
    featurehelper_x_test.add_feature(_tmp_tensor);

    let tmp_tensor = scaling.min_max_normalization_other(featurehelper_x_test.get_data_tensor());
    let prediction = model.predict(&tmp_tensor).unwrap();
    
    // 6 + 11 = 17 
    println!("Done {}", scaling.min_max_normalization_reverse( prediction.get(0).unwrap().clone() ) );
}

