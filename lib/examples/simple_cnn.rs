
#[allow(unused)]
use crate::prelude::*;
use crate::preprocessing::{self, features::{Features, FeaturesTrait}};

pub fn simple_cnn(){
    let varmap = VarMap::new();
    let dev = candle_core::Device::Cpu;

    let images = Tensor::read_npy("data/clock/clock_image.npy").unwrap();
    let results = Tensor::read_npy("data/clock/clock_time.npy").unwrap();

    let mut featurehelper_x = Features::new(dev.clone());
    let mut featurehelper_y = Features::new(dev.clone());

    let _images_vec: Vec<Vec<Vec<f32>>> = images.to_dtype(DType::F32).unwrap().to_vec3().unwrap();
    let _clock_vec: Vec<Vec<f32>> = results.to_dtype(DType::F32).unwrap().to_vec2().unwrap();
    for (_position, data) in _images_vec.iter().enumerate(){
        featurehelper_x.add_feature(Tensor::new(data.clone(), &dev).unwrap());
    }
    for (_position, data) in _clock_vec.iter().enumerate(){
        featurehelper_y.add_feature_1_d(data.clone());
    }

    let mut layers: Vec<Box<dyn Trainable>> = vec![];
    let mut name1 = String::new();
    name1.push_str("convolution 1");
    layers.push(Box::new(Conv::new2( ConvolutionTypes::Default, 2, 2, 1, 1, 1, &dev, &varmap, name1)));
    
    let mut name2 = String::new();
    name2.push_str("maxpooling 1");
    layers.push(Box::new(Pooling::new( PoolingType::MAX, 2, 2, &dev, &varmap, name2)));

    let mut name5 = String::new();
    name5.push_str("flatten");
    layers.push(Box::new(Flatten::new( &dev, &varmap, name5)));

    let mut name6 = String::new();
    name6.push_str("fully connected 1");
    layers.push(Box::new(Dense::new(8, 1024, Activations::Relu, &dev, &varmap, name6 )));

    let mut name7 = String::new();
    name7.push_str("fully connected 2");
    layers.push(Box::new(Dense::new(2, 8, Activations::Relu, &dev, &varmap, name7 )));
    
    let mut model = SequentialModel::new(varmap, layers); 

    let numbers: Vec<f32> = (0..=60).map(|x| x as f32).collect();
    let scaling = preprocessing::featurescaling::FeatureScaling::new(Tensor::new( numbers, &dev).unwrap());

    let tmp_x = featurehelper_x.get_data_tensor();
    let tmp_y = featurehelper_y.get_data_tensor();

    let x_testimage = images.get(0).unwrap().to_dtype(DType::F32).unwrap();
    let mut featurehelper_x_test = Features::new(dev.clone());
    featurehelper_x_test.add_feature(x_testimage);
    let x_testtensor = featurehelper_x_test.get_data_tensor();

    model.compile(Optimizers::SGD(0.0005), Loss::MSE);
    model.fit(
        scaling.min_max_normalization_other(tmp_x), 
        scaling.min_max_normalization_other(tmp_y), 
        4000, 
        false);


    let prediction = model.predict(&scaling.min_max_normalization_other(x_testtensor)).unwrap();
    println!("Prediction: {}", scaling.min_max_normalization_reverse( prediction.get(0).unwrap().clone() ));
    println!("Expected: {}", results.get(0).unwrap() );
}
