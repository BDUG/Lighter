#[allow(unused)]
use crate::prelude::*;
use crate::recurrenttypes::RecurrentType;

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
        1000, 
        true);
    
    let x_test: [[f32; 2]; 1] = [ [2., 1.] ];
    let prediction = model.predict(Tensor::new(&x_test, &dev).unwrap());
    println!("prediction: {}", prediction);
}
