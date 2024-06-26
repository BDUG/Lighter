

use std::collections::HashMap;

#[allow(unused)]
use candlelighter::prelude::*;
use candlelighter::saveweightstype::SaveWeightsType;

#[test]
fn save_model_test() -> anyhow::Result<()> {
    let varmap = VarMap::new();
    let dev = candle_core::Device::cuda_if_available(0).unwrap();  
    let _x: [[[f32; 2]; 1]; 6] = [ [[1., 2.]] , [[2., 1.]] ,[[3., 4.]], [[5., 6.]], [[5., 5.]] , [[4., 5.]]];
    let _y: [[[f32; 1]; 1]; 6] = [ [[3.]], [[3.]], [[7.]], [[11.]] , [[10.]], [[9.]]];

    let mut layers: Vec<Box<dyn Trainable>> = vec![];
    let mut name1 = String::new();
    name1.push_str("fc1");
    layers.push(Box::new(Dense::new(4, 2, Activations::Relu, &dev, &varmap, name1 )));  
    let mut name2 = String::new();
    name2.push_str("fc2");
    layers.push(Box::new(Dense::new(2, 4, Activations::Relu, &dev, &varmap, name2 )));
    let mut name3 = String::new();
    name3.push_str("fc3");
    layers.push(Box::new(Dense::new(1, 2, Activations::Relu, &dev, &varmap, name3 )));

    let model = SequentialModel::new(varmap.clone(), layers);

    model.save_model("./test.model");
    model.save_weights(SaveWeightsType::PlainJSON ,&varmap, "./test.weights");

    // Load the saved model and train

    let dev2 = candle_core::Device::cuda_if_available(0).unwrap();  
    let model2 = model.load_model("./test.model",&dev2);
    let mut hmap= HashMap::new();
    hmap.insert("path".into(), Value::String("./test.weights".into()));
    model2.load_weights(SaveWeightsType::PlainJSON ,&hmap, &varmap, &dev);

    anyhow::Ok(())
}


#[test]
fn summary_model_test() -> anyhow::Result<()> {
    let varmap = VarMap::new();
    let dev = candle_core::Device::cuda_if_available(0).unwrap();  
    let _x: [[[f32; 2]; 1]; 6] = [ [[1., 2.]] , [[2., 1.]] ,[[3., 4.]], [[5., 6.]], [[5., 5.]] , [[4., 5.]]];
    let _y: [[[f32; 1]; 1]; 6] = [ [[3.]], [[3.]], [[7.]], [[11.]] , [[10.]], [[9.]]];

    let mut layers: Vec<Box<dyn Trainable>> = vec![];
    let mut name1 = String::new();
    name1.push_str("fc1");
    layers.push(Box::new(Dense::new(4, 2, Activations::Relu, &dev, &varmap, name1 )));  
    let mut name2 = String::new();
    name2.push_str("fc2");
    layers.push(Box::new(Dense::new(2, 4, Activations::Relu, &dev, &varmap, name2 )));
    let mut name3 = String::new();
    name3.push_str("fc3");
    layers.push(Box::new(Dense::new(1, 2, Activations::Relu, &dev, &varmap, name3 )));

    let model = SequentialModel::new(varmap, layers);
    model.summary();

    anyhow::Ok(())
}
