
#[allow(unused)]
use crate::prelude::*;
use crate::recurrenttypes::RecurrentType;
use ndarray_rand::rand_distr::num_traits::ToPrimitive;
use rand::distributions::Distribution;
use crate::preprocessing;

/** This example implements initial parts of the steps given on the following site:
 *  https://github.com/javierlorenzod/pytorch-attention-mechanism
 *  https://github.com/philipperemy/keras-attention-mechanism/blob/master/examples/add_two_numbers.py
 */

struct TNNDataitem {
    x: Vec<usize>,
    y: usize
}

pub fn generatedata(sizeofsequence: usize, numofelements: usize, delimiter: f32) -> Vec<TNNDataitem> {
    let mut result: Vec<TNNDataitem> = vec![];

    let vals: Vec<u64> = (0..numofelements as u64).collect();
    for (i, _value) in vals.iter().enumerate() {
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
        for (j, value2) in vals2.iter().enumerate() {

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
    let numofelements = 2000;

    let varmap = VarMap::new();
    let dev = candle_core::Device::cuda_if_available(0).unwrap();
    let dataset = generatedata(sizeofsequence, numofelements, 0.0);

    let mut x: Vec<Vec<f32>> = Vec::new();
    let mut y: Vec<Vec<f32>> = Vec::new();

    for (_j, value) in dataset.iter().enumerate() {
        let tmp_value = value.x.iter().filter_map( |s| s.to_f32() ) .collect();
        x.push(tmp_value);

        let mut tmp_y: Vec<f32> = Vec::new();
        tmp_y.push(value.y.to_f32().unwrap());
        y.push(tmp_y);
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
    layers.push(Box::new(Dense::new(1, sizeofsequence, Activations::Relu, &dev, &varmap, name3 )));

    let mut model = SequentialModel::new(varmap, layers);
    model.compile(Optimizers::SGD(0.01), Loss::MSE);       

    let numbers: Vec<f32> = (0..=100).map(|x| x as f32).collect();
    let scaling = preprocessing::featurescaling::FeatureScaling::new(Tensor::new( numbers, &dev).unwrap());


    model.fit(
        scaling.min_max_normalization_other( to_tensor(&x,&dev).reshape((numofelements,1,sizeofsequence)).unwrap() ), // samples, _ , time steps
        scaling.min_max_normalization_other( to_tensor(&y,&dev).reshape((numofelements,1)).unwrap() ), // samples, _, expected calculated result
        1000, 
        false);
    
    let x_test: [[f32; 20]; 1] = [ [1., 2., 3., 4., 0., 6., 7., 8., 9., 0. ,11., 12.,12., 13., 14., 25., 16., 17., 18., 19., ] ];
    let prediction = model.predict( scaling.min_max_normalization_other( Tensor::new(&x_test, &dev).unwrap() ) );

    println!("Done {}", scaling.min_max_normalization_reverse( prediction) );
}

