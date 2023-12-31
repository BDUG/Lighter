use std::any::Any;

use crate::prelude::*;


pub struct Sequential {
    pub layers: Vec<Dense>,
    pub optimizer: Optimizers,
    pub loss: Loss,
}


impl Sequential {
    pub fn new(layers: &[Dense]) -> Self {
        Self {
            layers: layers.to_vec(),
            optimizer: Optimizers::None,
            loss: Loss::None,
        }
    }

    pub fn summary(&self) {
        // TODO
    }

    pub fn compile(&mut self, optimizer: Optimizers, loss: Loss) {
        self.optimizer = optimizer;
        self.loss = loss;
    }

    pub fn fit(&mut self, mut x: Tensor, _y: Tensor, epochs: usize ,_verbose: bool) {
        for _i in 0..epochs {
            if _verbose{
                println!("Epoche {}",_i);
            }
            // x.dims()[1]-1
            for elementnumber in 0.. 3 {
                let mut input_checked = match x.get(elementnumber) {
                    Ok(element) => element,
                    Err(error) => panic!("{}",error.to_string()),
                };

                let mut output_checked = match _y.get(elementnumber) {
                    Ok(element) => element,
                    Err(error) => panic!("{}",error.to_string()),
                };
                
                //let mut input_stack = vec![];
                for layer in self.layers.iter() {
                    input_checked = layer.forward(input_checked).clone();
                }
                // TODO: Apply loss
                let lossed =  match self.loss {
                    Loss::MSE => candle_nn::loss::mse(&input_checked, &output_checked),
                    Loss::NLL => candle_nn::loss::nll(&input_checked, &output_checked),
                    Loss::BinaryCrossEntropyWithLogit => candle_nn::loss::binary_cross_entropy_with_logit(&input_checked, &output_checked),
                    Loss::CrossEntropy => candle_nn::loss::cross_entropy(&input_checked, &output_checked),
                    Loss::None => todo!(),
                };

                let lossed_checked = match lossed {
                    Ok(lossed) => lossed,
                    Err(error) => panic!("{}",error.to_string()),
                };
                output_checked = lossed_checked.clone();

                let enumvalue: u8 = match self.optimizer {
                    Optimizers::SGD =>  1,
                    Optimizers::Adam => 2,
                    Optimizers::None => 0,
                };
                // Apply optimizer 
                let lossed_backup = lossed_checked.clone();
                let varmap = VarMap::new();
                if enumvalue == 1 {
                    let mut optimized: SGD = candle_nn::SGD::new(varmap.all_vars(), 0.01).unwrap();
                    optimized.backward_step(&lossed_checked);
                }
                else if enumvalue == 2 {
                    let adamw_params = candle_nn::ParamsAdamW {
                        lr: 0.0001,
                        ..Default::default()
                    };
                    let mut optimized: AdamW = candle_nn::AdamW::new(varmap.all_vars(), adamw_params).unwrap();
                    optimized.backward_step(&lossed_checked);
                }
            }
        }
    }


    pub fn predict(&self, mut x: Tensor) -> Tensor {
        for layer in self.layers.iter() {
            x = layer.forward(x);
        }
        return x;
    }

    /*
    pub fn save(&self, path: &str) {
        let encoded: Vec<u8> = bincode::serialize(&self.layers).unwrap();
        let mut file = File::create(path).unwrap();
        file.write(&encoded).unwrap();
    }

    pub fn load(&self, path: &str) -> Sequential<Dense>{
        let mut file = File::open(path).unwrap();
        let mut decoded = Vec::new();
        file.read_to_end(&mut decoded).unwrap();
        let model: Sequential<_> = bincode::deserialize(&decoded[..]).unwrap();
        println!("model: {:?}", model);
        model
    }
     */

}

