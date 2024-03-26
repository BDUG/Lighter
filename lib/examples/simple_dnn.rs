use rand::distributions::Distribution;

#[allow(unused)]
use crate::prelude::*;
use crate::preprocessing;

struct Dataitem {
    x: Vec<usize>,
    y: usize
}

pub fn generatedata(numofelements: usize, limit: usize) -> Vec<Dataitem> {
    let mut result: Vec<Dataitem> = vec![];

    let vals: Vec<u64> = (0..numofelements as u64).collect();
    for (i, value) in vals.iter().enumerate() {
        let index = Uniform::new(0, limit);
        let mut rng = rand::thread_rng();
        let a = index.sample(&mut rng);
        let b = index.sample(&mut rng);


        let mut resultelement =  Dataitem {
            x: Vec::new(),
            y : a+b
        };
        resultelement.x.push(a);
        resultelement.x.push(b);

        result.push(resultelement);
    }
    return result;
}


pub fn simple_dnn() {
    let varmap = VarMap::new();
    let dev = candle_core::Device::cuda_if_available(0).unwrap();


    let dataset = generatedata(2000, 10);

    let mut _x: Vec<Vec<f32>> = Vec::new();
    let mut _y: Vec<Vec<f32>> = Vec::new();

    for (j, value) in dataset.iter().enumerate() {
        let tmp_value = value.x.iter().filter_map( |s| Some(*s as f32) ) .collect();
        _x.push(tmp_value);

        let mut tmp_y: Vec<f32> = Vec::new();
        tmp_y.push(value.y as f32);
        _y.push(tmp_y);
    }
    let mut _x1layer: Vec<_> = Vec::new();
    let mut _x2layer: Vec<_> = Vec::new();
    _x2layer.push(_x);
    _x1layer.push(_x2layer);
    let mut _y1layer: Vec<_>= Vec::new();
    let mut _y2layer: Vec<_> = Vec::new();
    _y2layer.push(_y);
    _y1layer.push(_y2layer);

    
    //let x: [[[f32; 2]; 1]; 6] = [ [[1., 2.]] , [[2., 1.]] ,[[3., 4.]], [[5., 6.]], [[5., 5.]] , [[4., 5.]]];
    //let y: [[[f32; 1]; 1]; 6] = [ [[3.]], [[3.]], [[7.]], [[11.]] , [[10.]], [[9.]]];

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
    
    //let tmp_x = Tensor::new(&x, &dev).unwrap();
    //let tmp_y = Tensor::new(&y, &dev).unwrap();
    let tmp_x = Tensor::new(_x1layer, &dev).unwrap();
    let tmp_y = Tensor::new(_y1layer, &dev).unwrap();

    let numbers: Vec<f32> = (0..=100).map(|x| x as f32).collect();
    let scaling = preprocessing::featurescaling::FeatureScaling::new(Tensor::new( numbers, &dev).unwrap());

    model.compile(Optimizers::SGD(0.001), Loss::MSE);
    model.fit(
        scaling.min_max_normalization_other(tmp_x), 
        scaling.min_max_normalization_other(tmp_y), 
        3000, 
        false);
    
    let x_test: [[f32; 2]; 1] = [ [4., 5.] ];
    let tmp_tensor = scaling.min_max_normalization_other(Tensor::new(&x_test, &dev).unwrap());
    let prediction = model.predict(tmp_tensor);
    println!("Prediction: {}", scaling.min_max_normalization_reverse(prediction));
}
