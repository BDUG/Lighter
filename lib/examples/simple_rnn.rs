use std::ops::Add;

#[allow(unused)]
use crate::prelude::*;
use crate::recurrenttypes::RecurrentType;
use rand::Rng;

pub fn simple_rnn() {
    let varmap = VarMap::new();
    let dev = candle_core::Device::cuda_if_available(0).unwrap();
    let x: [[[f32; 2]; 1]; 6] = [ [[1., 2.]] , [[2., 1.]] ,[[3., 4.]], [[5., 6.]], [[5., 5.]] , [[4., 5.]]];
    let y: [[[f32; 1]; 1]; 6] = [ [[3.]], [[3.]], [[7.]], [[11.]] , [[10.]], [[9.]]];

    let mut layers: Vec<Box<dyn Trainable>> = vec![];
    let mut name1 = String::new();
    name1.push_str("rnn1");
    layers.push(Box::new(Recurrent::new(RecurrentType::LSTM,2, 4, &dev, &varmap, name1 )));
    let mut name3 = String::new();
    name3.push_str("fc1");
    layers.push(Box::new(Dense::new(1, 4, Activations::Relu, &dev, &varmap, name3 )));

    let mut model = SequentialModel::new(varmap, layers);
    
    model.compile(Optimizers::SGD(0.01), Loss::MSE);
    model.fit(
        Tensor::new(&x, &dev).unwrap(), 
        Tensor::new(&y, &dev).unwrap(), 
        2000, 
        true);
    
    let x_test: [[f32; 2]; 1] = [ [1., 1.] ];
    let prediction = model.predict(&Tensor::new(&x_test, &dev).unwrap()).unwrap();
    println!("prediction: {}", prediction.get(0).unwrap().clone() );
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

pub fn generate_sum_pair(batchsize: usize) -> Vec<Vec<Vec<f32>>> {
    let mut rng = rand::thread_rng();

    let mut input_vector : Vec<Vec<f32>> = Vec::new();
    let mut output_vector : Vec<Vec<f32>> = Vec::new();

    for _i in 0..batchsize {
        let n1: f32 = rng.gen_range(0.0..100.0);
        let n2: f32 = rng.gen_range(0.0..100.0);
        let mut input_pair : Vec<f32> = Vec::new();
        input_pair.push(n1);
        input_pair.push(n2);
    
        let sum: f32 = n1.add(n2);
        let mut output_pair : Vec<f32> = Vec::new();
        output_pair.push(sum);

        input_vector.push(input_pair);
        output_vector.push(output_pair);
        
    }

    let mut result: Vec<Vec<Vec<f32>>> = Vec::new();
    result.push(input_vector);
    result.push(output_vector);
    return result;
}


// TBD: DO addition with given textual description e.g. 1+1
pub fn simple_rnn2() {
    let varmap = VarMap::new();
    let dev = candle_core::Device::cuda_if_available(0).unwrap();
    let generateddata: Vec<Vec<Vec<f32>>> = generate_sum_pair(100);
    let x_tmp: &Vec<Vec<f32>> = generateddata.get(0).unwrap();
    let y_tmp: &Vec<Vec<f32>> = generateddata.get(1).unwrap();

    let _xsize : usize = x_tmp.len();
    let x = to_tensor(x_tmp, &dev);
    let y = to_tensor(y_tmp, &dev);

    let mut layers: Vec<Box<dyn Trainable>> = vec![];
    let mut name1 = String::new();
    name1.push_str("rnn1");
    layers.push(Box::new(Recurrent::new(RecurrentType::LSTM,2, 4, &dev, &varmap, name1 )));
    let mut name3 = String::new();
    name3.push_str("fc1");
    layers.push(Box::new(Dense::new(1, 4, Activations::Relu, &dev, &varmap, name3 )));

    let mut model = SequentialModel::new(varmap, layers);
    model.compile(Optimizers::SGD(0.01), Loss::MSE);
    model.fit(
        x, 
        y, 
        2000, 
        true);
    
    let x_test: [[f32; 2]; 1] = [ [2., 1.] ];
    let prediction = model.predict(&Tensor::new(&x_test, &dev).unwrap()).unwrap();
    println!("prediction: {}", prediction.get(0).unwrap().clone() );
}