
#[allow(unused)]
use crate::prelude::*;

pub fn simple_cnn(){
    let varmap = VarMap::new();
    let dev = candle_core::Device::Cpu;//cuda_if_available(0).unwrap();

    //--------- JUST for testing
    /*
    let x: [[f32; 1]; 2] = [ [2.6263], [1.1000] ];
    let y: [u32; 2] = [8, 5];

    let a = Tensor::new(&x, &dev).unwrap();
    let b = Tensor::new(&y, &dev).unwrap();

    let rst = candle_nn::loss::nll(&a, &b);
     */
    //---------

    let images = Tensor::read_npy("data/clock/clock_image.npy").unwrap();
    let results = Tensor::read_npy("data/clock/clock_time.npy").unwrap();

    let x_train = images.clone();
    let x_test = images.clone();
    let y_test = results.clone();
    let x_testimage = x_test.get(0).unwrap();
    let y_result = y_test.get(0).unwrap();

    let mut layers: Vec<Box<dyn Trainable>> = vec![];
    let mut name1 = String::new();
    name1.push_str("convolution 1");
    layers.push(Box::new(Conv::new2( ConvolutionTypes::Default, 2, 2, 2, 1, 1, &dev, &varmap, name1)));
    
    let mut name2 = String::new();
    name2.push_str("maxpooling 1");
    layers.push(Box::new(Pooling::new( PoolingType::AVERAGE, 2, 2, &dev, &varmap, name2)));

    let mut name3 = String::new();
    name3.push_str("convolution 2");
    layers.push(Box::new(Conv::new2( ConvolutionTypes::Default, 2, 2, 1, 1, 1, &dev, &varmap, name3)));

    let mut name4 = String::new();
    name4.push_str("maxpooling 2");
    layers.push(Box::new(Pooling::new( PoolingType::AVERAGE, 2, 2, &dev, &varmap, name4)));

    let mut name5 = String::new();
    name5.push_str("flatten");
    layers.push(Box::new(Flatten::new( &dev, &varmap, name5)));

    let mut name6 = String::new();
    name6.push_str("fully connected 1");
    layers.push(Box::new(Dense::new(8, 256, Activations::Relu, &dev, &varmap, name6 )));

    let mut name7 = String::new();
    name7.push_str("fully connected 2");
    layers.push(Box::new(Dense::new(2, 8, Activations::Relu, &dev, &varmap, name7 )));
    
    let mut model = SequentialModel::new(varmap, layers); 
    model.compile(Optimizers::SGD(0.00001), Loss::MSE);

    model.fit(
        x_train, 
        results, 
        10, 
        true);

    let prediction = model.predict(x_testimage);
    println!("prediction: {} \n vs. {}", prediction, y_result);
    println!("Done");
}
