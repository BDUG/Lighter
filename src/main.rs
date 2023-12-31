use lighter::prelude::*;

fn main() {
    let dev = candle_core::Device::cuda_if_available(0).unwrap();  
    let x: [[[f32; 2]; 1]; 6] = [ [[1., 2.]] , [[2., 1.]] ,[[3., 4.]], [[5., 6.]], [[5., 5.]] , [[4., 5.]]];
    let y: [[[f32; 1]; 1]; 6] = [ [[3.]], [[3.]], [[7.]], [[11.]] , [[10.]], [[9.]]];

    let mut model = lighter::models::Sequential::new(&[
        Dense::new(4, 2, &dev, Activations::Relu),
        Dense::new(2, 4, &dev, Activations::Relu),
        Dense::new(1, 2, &dev, Activations::Relu),
    ]);
    
    // TODO learnign rate 0.01
    model.compile(Optimizers::SGD, Loss::MSE);
    model.fit(
        Tensor::new(&x, &dev).unwrap(), 
        Tensor::new(&y, &dev).unwrap(), 
        10000, 
        true);
    
    let x_test: [[f32; 2]; 1] = [ [2., 1.] ];
    let prediction = model.predict(Tensor::new(&x_test, &dev).unwrap());
    println!("prediction: {}", prediction);

    //model.save("./test.model");
}
