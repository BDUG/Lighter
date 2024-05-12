

#[allow(unused)]
pub use crate::prelude::*;

pub struct Features {
     //
    // Example:
    // let x: [[[f32; 2]; 1]; 6] = [ [[1., 2.]] , [[2., 1.]] ,[[3., 4.]], [[5., 6.]], [[5., 5.]] , [[4., 5.]]];
    //
    // Array structure is [[[f32; Spatial]; Temporal]; Rational] 
    //
    data: Vec< // Rational
                    Vec< // Temporal
                        Tensor> // Spatial
                    >,
    device: Device 
}

pub trait FeaturesTrait {
    fn new(dev:Device) -> Self;
    fn add_temporal_feature(&mut self, input: Vec<Tensor>);
    fn add_feature(&mut self, input: Tensor);
    fn add_temporal_feature_1_d(&mut self, input: Vec<Vec<f32>>);
    fn add_feature_1_d(&mut self, input: Vec<f32>);
    fn get_amount_of_sample(&mut self) -> usize;
    fn get_amount_of_temporal_steps(&mut self) -> usize;
    fn get_data_tensor(&mut self) ->Tensor;
}

impl FeaturesTrait for Features {
    fn new(dev: Device) -> Self {
        let mut _container: Vec<_> = Vec::new();
        Self {
            data: _container, // Put it on the heap
            device: dev
        }
    }
    
    fn add_temporal_feature(&mut self, input: Vec<Tensor>) {
        self.data.push(input.clone());
    }
    
    fn add_feature(&mut self, input: Tensor) {
        let mut _temporal: Vec<_> = Vec::new();
        _temporal.push( input );
        self.data.push(_temporal.clone());
    }
    
    fn get_amount_of_sample(&mut self) -> usize {
        return self.data.len();
    }
    
    fn get_amount_of_temporal_steps(&mut self) -> usize {
        let _tmp = self.data.get(0).unwrap();
        return _tmp.len();
    }
    
    fn add_temporal_feature_1_d(&mut self, input: Vec<Vec<f32>>) {
        let mut _temporal: Vec<Tensor> = Vec::new();
        _temporal.push( Tensor::new(input, &self.device ).unwrap() );

       self.add_temporal_feature(_temporal);
    }
    
    fn add_feature_1_d(&mut self, input: Vec<f32>) {
       self.add_feature(Tensor::new(input, &self.device).unwrap());
    }
    
    fn get_data_tensor(&mut self) ->Tensor {
        let mut _rational: Vec<Vec<Vec<f32>>> = Vec::new();
        let mut _dimensions = 0;
        let mut _dim = None;
        for (_position1, data1) in self.data.iter().enumerate(){
            let mut _temporal: Vec<Vec<f32>> = Vec::new();
            for (_position2, data2) in data1.iter().enumerate(){
                if _dimensions == 0{
                    _dim = Some(data2.shape().dims());
                    _dimensions = data2.shape().dims().len();
                    if _dimensions >3 {
                        todo!();
                    }
                }
                let data3 = data2.to_dtype(DType::F32).unwrap();
                _temporal.push( data3.flatten_all().unwrap().to_vec1::<f32>().unwrap() );
            }
            _rational.push(_temporal);
        }
        let _result: Tensor = Tensor::new(_rational, &self.device ).unwrap();
        let mut _new_shape: Vec<usize> = Vec::new();
        for i in _result.shape().dims(){
            _new_shape.push(*i);
        }
        _new_shape.pop();
        for i in _dim.unwrap(){
            _new_shape.push(*i);
        }

        return _result.reshape(_new_shape).unwrap().clone();

    }
    
   
}