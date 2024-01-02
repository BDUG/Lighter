use lighter::prelude::*;

fn main() {
    let varmap = VarMap::new();
    let dev = candle_core::Device::cuda_if_available(0).unwrap();  
    let x: [[[f32; 2]; 1]; 6] = [ [[1., 2.]] , [[2., 1.]] ,[[3., 4.]], [[5., 6.]], [[5., 5.]] , [[4., 5.]]];
    let y: [[[f32; 1]; 1]; 6] = [ [[3.]], [[3.]], [[7.]], [[11.]] , [[10.]], [[9.]]];

    let mut layers = vec![];
    let mut name1 = String::new();
    name1.push_str("fc1");
    layers.push(Dense::new(4, 2, Activations::Relu, &dev, &varmap, name1 ));  
    let mut name2 = String::new();
    name2.push_str("fc2");
    layers.push(Dense::new(2, 4, Activations::Relu, &dev, &varmap, name2 ));
    let mut name3 = String::new();
    name3.push_str("fc3");
    layers.push(Dense::new(1, 2, Activations::Relu, &dev, &varmap, name3 ));

    let mut model = lighter::models::Sequential::new(varmap, layers);
    
    // TODO learnign rate 0.01
    model.compile(Optimizers::SGD, Loss::MSE);
    model.fit(
        Tensor::new(&x, &dev).unwrap(), 
        Tensor::new(&y, &dev).unwrap(), 
        1000, 
        true);
    
    let x_test: [[f32; 2]; 1] = [ [2., 1.] ];
    let prediction = model.predict(Tensor::new(&x_test, &dev).unwrap());
    println!("prediction: {}", prediction);

    // Save the model

    model.save_model("./test.model");

    // Load the saved model and train

    let dev2 = candle_core::Device::cuda_if_available(0).unwrap();  
    let mut model2 = model.load_model("./test.model",&dev2);
    model2.fit(
        Tensor::new(&x, &dev2).unwrap(), 
        Tensor::new(&y, &dev2).unwrap(), 
        1000, 
        false);

    let prediction2 = model2.predict(Tensor::new(&x_test, &dev2).unwrap());
    println!("prediction: {}", prediction2);
 
}
