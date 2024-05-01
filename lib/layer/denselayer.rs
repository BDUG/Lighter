

use std::ops::Div;

use crate::densetypes::DenseType;
use candle_core::Shape;
#[allow(unused)]
pub use crate::prelude::*;

pub struct Dense {
    pub activation: Activations,
    pub perceptrons: usize,
    pub previousperceptrons: usize,
    pub denselayer: Linear,
    pub device: Device,
    pub types: DenseType,
    pub a: Tensor,
    pub b: Tensor,
    pub alpha: f32,
    pub name: String,
}

pub trait DenseLayerTrait {
    fn new(perceptrons: usize, previousperceptrons: usize, activation: Activations, device: &Device, varmap : &VarMap, name: String) -> Self;
    fn new2(perceptrons: usize, previousperceptrons: usize, activation: Activations, types:DenseType , rank: Tensor, alpha: f32, device: &Device, varmap : &VarMap, name: String) -> Self;
}

impl DenseLayerTrait for Dense {
    fn new(perceptrons: usize, previousperceptrons: usize, activation: Activations, device: &Device, varmap : &VarMap, name: String) -> Self {
        let vs = VarBuilder::from_varmap(varmap, DType::F32, &device);
        let tmp_name = name.clone();
        Self {
            activation : activation,
            perceptrons : perceptrons,
            previousperceptrons : previousperceptrons,
            denselayer : linear(previousperceptrons, perceptrons,vs.pp(name)).unwrap(),
            device : device.clone(),
            types: DenseType::Standard,
            a: Tensor::zeros(Shape::from_dims(&[0, 0]), DType::F32, device).unwrap(),
            b: Tensor::zeros(Shape::from_dims(&[0, 0]), DType::F32, device).unwrap(),
            alpha: 0.0,
            name: tmp_name.clone(),
        }
    }

    fn new2(perceptrons: usize, previousperceptrons: usize, activation: Activations, types: DenseType, rank: Tensor, alpha: f32, device: &Device, varmap : &VarMap, name: String) -> Self {
        if types.eq(&DenseType::Standard) {
            panic!("Use new instead of new2 for standard dense layer");
        }
        let vs = VarBuilder::from_varmap(varmap, DType::F32, &device);
        let tmp_name = name.clone();

        let mut _a:Option< Tensor > = None;
        let mut _b:Option< Tensor > = None;
        // Fine tune the layer
        if types.eq(&DenseType::LORA) || types.eq(&DenseType::DORA) {
            let std_dev = ( (1.0 as f32).div( (rank.rank() as f32 ).sqrt()) ) as f32;
            let _rank = rank.rank();

            let _tmp =  Tensor::rand(0.0, 1.0, (previousperceptrons,_rank), device).unwrap().to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let _tmp2: Vec<f32>  = _tmp.iter().map( |x| x * std_dev).collect::<Vec<f32>>();
            _a = Some(Tensor::new(_tmp2, device).unwrap().reshape(Shape::from_dims(&[_rank,previousperceptrons])).unwrap());
            _b = Some(Tensor::zeros(Shape::from_dims(&[_rank,previousperceptrons]), DType::F32, device).unwrap());
        }

        Self {
            activation : activation,
            perceptrons : perceptrons,
            previousperceptrons : previousperceptrons,
            denselayer : linear(previousperceptrons, perceptrons,vs.pp(name)).unwrap(),
            device : device.clone(),
            types: types,
            a: _a.unwrap(),
            b: _b.unwrap(),
            alpha : alpha,
            name: tmp_name.clone(),
        }
    }

}


impl Trainable for Dense {
    
    // .to_scalar::<f64>()
    fn forward(&self,  input : Tensor) -> Tensor {
        // Apply layer calculation
        let new_tensor = input.to_dtype(DType::F32).unwrap();

        let mut fullyconnected_checked = None;

        if self.types.eq(&DenseType::Standard) {
            let fullyconnected = self.denselayer.forward(&new_tensor);
            fullyconnected_checked = match fullyconnected {
                Ok(fullyconnected) => Some(fullyconnected),
                Err(error) => panic!("{}",error.to_string()),
            };
        }
        else if self.types.eq(&DenseType::LORA) || self.types.eq(&DenseType::DORA) {
            // LORA fine tune
            let _tmp = ( new_tensor.matmul( &self.a.matmul(&self.b).unwrap() )).unwrap();
            let _tmp2 = _tmp.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let _tmp3: Vec<f32>  = _tmp2.iter().map( |x| x * self.alpha).collect::<Vec<f32>>();
            let _tmp4 = &Tensor::new(_tmp3, input.device()).unwrap().reshape(new_tensor.shape()).unwrap();

            fullyconnected_checked = Some( self.denselayer.forward(_tmp4).unwrap() );

            if self.types.eq(&DenseType::DORA) {
                // FIX ME

                // L2 Norm = Euclidian
                let denominator =  fullyconnected_checked.clone().unwrap().broadcast_div(& (fullyconnected_checked.clone().unwrap().sqr().unwrap().sum_keepdim(1).unwrap().sqrt().unwrap() ) ).unwrap();
                let directional_component = (fullyconnected_checked.unwrap().div(denominator) ).unwrap();

                // L2 Norm = Euclidian
                let m =  self.denselayer.weight().broadcast_div(&self.denselayer.weight().sqr().unwrap().sum_keepdim(1).unwrap().sqrt().unwrap()).unwrap();
                let new_weight = m.t().unwrap().matmul(&directional_component.t().unwrap()).unwrap();

                self.denselayer.weight().clone_from(&&new_weight);             

                fullyconnected_checked = Some( self.denselayer.forward(&new_tensor).unwrap() );
            }
        }
        
        // Apply activation
        let activated = match self.activation {
            Activations::Linear => Ok(fullyconnected_checked.unwrap().clone()),
            Activations::Relu => fullyconnected_checked.unwrap().clone().relu(),
            Activations::Silu => ops::silu(&fullyconnected_checked.unwrap().clone()),
            Activations::Sigmoid => ops::sigmoid(&fullyconnected_checked.unwrap().clone()),
            Activations::Softmax => ops::log_softmax(&fullyconnected_checked.unwrap().clone(), D::Minus1),
        };
        let activated_checked = match activated {
            Ok(activated) => activated,
            Err(error) => panic!("{}",error.to_string()),
        };
        return activated_checked;
    }

    fn typ(&self) -> String {
        "Dense".into()
    }

    fn input_perceptrons(&self) -> u32{
        return self.previousperceptrons as u32;
    }
    fn output_perceptrons(&self) -> u32{
        return self.perceptrons as u32;
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
    
}

