
use flatten::embeddinglayer::{Embed, EmbeddingLayerTrait};
use layer::sparsemoe::{SparseMoE, SparseMoETrait};
use ndarray_rand::rand_distr::num_traits::ToPrimitive;

use crate::layer;
#[allow(unused)]
use crate::prelude::*;
use crate::recurrenttypes::RecurrentType;
use crate::embeddingtypes::EmbeddingType;


pub trait ModelSerialization {

    fn extractjson_value_u64(&self, elem: &serde_json::Map<String, Value>, key: &str) -> u64{
        let value: Option<(&String, &Value)> = elem.get_key_value(key);
        let rst = match value {
            Some(x) => x.1.clone(),
            None    => panic!("Unknown state"),
        };
        if rst.is_u64() {
            return u64::from(rst.as_u64().unwrap());
        }
        panic!("Unknown type");
    }

    fn extractjson_value_f64(&self, elem: &serde_json::Map<String, Value>, key: &str) -> f64{
        let value: Option<(&String, &Value)> = elem.get_key_value(key);
        let rst = match value {
            Some(x) => x.1.clone(),
            None    => panic!("Unknown state"),
        };
        if rst.is_f64() {
            return f64::from(rst.as_f64().unwrap());
        }
        panic!("Unknown type");
    }


    fn extractjson_value_serializedtensor(&self, elem: &serde_json::Map<String, Value>, key: &str, device: &Device) -> Tensor{
        let value: Option<(&String, &Value)> = elem.get_key_value(key);
        let _rst = match value {
            Some(x) => x.1.clone(),
            None    => panic!("Unknown state"),
        };
        let _serializedtensormap = value.unwrap().1.as_object().unwrap();
        let tmp_dimension = _serializedtensormap.get("dimension").unwrap().as_array().unwrap().to_vec();
        let tmp_values = _serializedtensormap.get("values").unwrap().as_array().unwrap().to_vec();
        let dimension = tmp_dimension.iter().map(|x| x.as_f64().unwrap() as usize).collect::<Vec<usize>>();
        let values = tmp_values.iter().map(|x| x.as_f64().unwrap() as f32).collect::<Vec<f32>>();
        

        let mut resulttensor = Tensor::new(values, device).unwrap();
        if dimension.len() > 1{        
            resulttensor = resulttensor.clone().reshape(dimension.as_slice()).unwrap();
        }
        return resulttensor;
    }

    fn extractjson_value_str(&self, elem: &serde_json::Map<String, Value>, key: &str)-> String {
        let value: Option<(&String, &Value)> = elem.get_key_value(key);
        let rst = match value {
            Some(x) => x.1.clone(),
            None    => panic!("Unknown state"),
        };
        if rst.is_string(){
            return String::from(rst.as_str().unwrap());
        }
        panic!("Unknown type");
    }

    fn save_weights_generic(&self, varmap: &VarMap, path: &str) {
        let mut toserialize : Vec<SerializedTensor> = Vec::new();

        let tmp = varmap.data();
        let data = tmp.lock().unwrap();
        for element in data.keys(){
            let raw_tensor = data.get_key_value(element).unwrap().1;
            let tensor = raw_tensor.flatten_all().unwrap();
            let ser_tensor = SerializedTensor {
                name: element.clone(),
                dimension : raw_tensor.shape().dims().to_vec(),
                values : tensor.to_vec1::<f32>().unwrap(),
            };
            toserialize.push(ser_tensor);
        }
      
        let mut file = File::create(path).unwrap();
        let j = serde_json::to_string(&toserialize).unwrap();
        file.write(&j.as_bytes()).unwrap();
    }

    fn load_weights_generic(&self, path: &str, varmap: &VarMap, device: &Device) { // -> VarMap
        //let varmap: VarMap = VarMap::new();
        varmap.data().lock().unwrap().clear();
        let value = fs::read_to_string(path).unwrap();

        let json: serde_json::Value =
            serde_json::from_str(&value).unwrap();
        
        let a: Vec<Value> = json.as_array().unwrap().to_vec();
        for i in 0..a.len(){
            let b = a.get(i).unwrap();
            let c = b.as_object().unwrap();
            let mut d = c.values().rev();
            let _values = d.next().unwrap().as_array();
            let _name = d.next().unwrap().as_str();
            let _dimension = d.next().unwrap().as_array();
            let values = match _values {
                Some(x) => x.clone().iter().map(|x| x.as_f64().unwrap() as f32).collect::<Vec<f32>>(),
                None    => panic!("Unknown state"),
            };
            let name = match _name {
                Some(x) => x, //.clone(),
                None    => panic!("Unknown state"),
            };
            varmap.data().lock().unwrap().insert(name.to_string(),Var::new(values, &device).unwrap());
        }
       // return varmap;
    }

    fn load_layer (&self, new_type: String, elem: &serde_json::Map<String, Value>, device: &Device, layers: &mut Vec<Box<dyn Trainable>>, varmap: &VarMap) {
        if new_type.eq("Dense"){
            let new_perceptrons = self.extractjson_value_u64(elem, "perceptrons");
            let new_previousperceptrons = self.extractjson_value_u64(elem, "previousperceptrons");
            let new_name = self.extractjson_value_str(elem, "name");
            let tmp_activation = self.extractjson_value_str(elem, "activation");
            let new_activation = Activations::from_string(tmp_activation.to_string());
    
            let tmp = Box::new(Dense::new(new_perceptrons as usize, new_previousperceptrons as usize, new_activation, device, varmap, new_name));
            layers.push(tmp);
        }
        else if new_type.eq("Pooling"){
            let new_poolingtype = self.extractjson_value_str(elem, "poolingtype");
            let new_kernelsize = self.extractjson_value_u64(elem, "kernelsize") as usize;
            let new_stride = self.extractjson_value_u64(elem, "stride") as usize;
            let new_name = self.extractjson_value_str(elem, "name");
        
            let tmp = Box::new(Pooling::new(PoolingType::from_string(new_poolingtype), new_kernelsize, new_stride as usize, &device, varmap, new_name));
            layers.push(tmp);
        }
        else if new_type.eq("Recurrent"){
            let new_recurrenttype = self.extractjson_value_str(elem, "recurrenttype");
            let new_indimension = self.extractjson_value_u64(elem, "indimension") as usize;
            let new_hiddendimension = self.extractjson_value_u64(elem, "hiddendimension") as usize;
            let new_name = self.extractjson_value_str(elem, "name");
        
            let tmp = Box::new(Recurrent::new(RecurrentType::from_string(new_recurrenttype), new_indimension, new_hiddendimension, &device, varmap, new_name));
            layers.push(tmp);
        }
        else if new_type.eq("Embedding"){
            let new_inputdimension = self.extractjson_value_u64(elem, "inputdimension") as usize;
            let new_hiddendimension = self.extractjson_value_u64(elem, "hiddendimension") as usize;
            let new_name = self.extractjson_value_str(elem, "name");
        
            let tmp = Box::new(Embed::new(EmbeddingType::Standard, new_inputdimension, new_hiddendimension, &device, varmap, new_name));
            layers.push(tmp);
        }
        else if new_type.eq("Normalization"){
            let new_axis = self.extractjson_value_u64(elem, "axis") as u64;
            let new_name = self.extractjson_value_str(elem, "name");
        
            let tmp = Box::new(Normalization::new( new_axis, &device, varmap, new_name));
            layers.push(tmp);
        }
        else if new_type.eq("Conv"){
            let new_tensor = self.extractjson_value_serializedtensor(elem, "kernel", device);
            let new_dimensionality = self.extractjson_value_u64(elem, "dimensionality") as usize;
            let new_padding = self.extractjson_value_u64(elem, "padding") as usize;
            let new_stride = self.extractjson_value_u64(elem, "stride") as usize;
            let new_dilation = self.extractjson_value_u64(elem, "dilation") as usize;
            let new_groups = self.extractjson_value_u64(elem, "groups") as usize;
            let new_name = self.extractjson_value_str(elem, "name");
    
            let tmp = Box::new(Conv::new(new_tensor, new_dimensionality,new_padding, new_stride, new_dilation, new_groups,device, varmap,new_name));
            layers.push(tmp);
        }
        else if new_type.eq("SelfAttention"){
            let new_query_dim = self.extractjson_value_u64(elem, "query_dim") as usize;
            let new_heads = self.extractjson_value_u64(elem, "heads") as usize;
            let new_dim_head = self.extractjson_value_u64(elem, "dim_head") as usize;
            let new_input_size = self.extractjson_value_u64(elem, "input_size") as usize;
            let new_name = self.extractjson_value_str(elem, "name");
    
            let tmp = Box::new(SelfAttention::new(new_query_dim, new_heads, new_dim_head, new_input_size, device, varmap,new_name));
            layers.push(tmp);
        }
        else if new_type.eq("SparseMoE"){
            let new_num_of_experts = self.extractjson_value_u64(elem, "num_of_experts") as usize;
            let new_input_dim = self.extractjson_value_u64(elem, "input_dim") as usize;
            let new_output_dim = self.extractjson_value_u64(elem, "output_dim") as usize;
            let new_name = self.extractjson_value_str(elem, "name");
    
            let tmp = Box::new(SparseMoE::new(new_num_of_experts, new_input_dim, new_output_dim, device, varmap,new_name));
            layers.push(tmp);
        }
        else{
            panic!("Unknown layer type {}",new_type.to_string())
        }
    }

}


/// A trait for objects that can make predictions based on input data.
pub trait Predictable {

    /// Predicts the output for the given input data using the provided layers.
    ///
    /// # Arguments
    ///
    /// * `layers` - A vector of trainable layers to be used for prediction.
    /// * `x` - The input data tensor.
    ///
    /// # Returns
    ///
    /// An optional vector of output tensors representing the predictions. If the prediction fails,
    /// `None` is returned.
    fn predicting(&self, layers: &Vec<Box<dyn Trainable>> , x: &Tensor) -> Option<Vec<Tensor>> {
        let mut result: Vec<Tensor> = Vec::new();
        for elementnumber in 0.. x.dims().get(0).unwrap().to_usize().unwrap() {
            let mut _original_data = x.clone();
            let mut input_checked = match _original_data.get(elementnumber) {
                Ok(element) => element,
                Err(error) => panic!("{}",error.to_string()),
            };
            for layer in layers.iter() {
                input_checked = layer.forward(input_checked);
            }
            result.push(input_checked);
        }
        return Some(result);
    }
}

pub trait Fitable {

    fn fitting(&self, layers: &Vec<Box<dyn Trainable>> ,loss: &Loss, optimizer: &Optimizers,varmap: &VarMap, epochs: usize, _verbose: bool, x: &Tensor, _y: &Tensor) ->VarMap  {  
        let mut bestloss = f32::MAX;
        let mut snapshot:Option< VarMap > = None;
        for _i in 0..epochs {
            if _verbose{
                println!("Epoche {}",_i);
            }
            for elementnumber in 0.. x.dims().get(0).unwrap().to_usize().unwrap() {
                let mut input_checked = match x.get(elementnumber) {
                    Ok(element) => element,
                    Err(error) => panic!("{}",error.to_string()),
                };
        
                let mut output_checked = match _y.get(elementnumber) {
                    Ok(element) => element,
                    Err(error) => panic!("{}",error.to_string()),
                };
        
                for layer in layers.iter() {
                    input_checked = layer.forward(input_checked).clone();
                }
                if input_checked.shape().dims().len() == 1 {
                    input_checked = input_checked.reshape((1, input_checked.shape().dims().get(0).unwrap().to_owned())).unwrap();
                }
                if output_checked.shape().dims().len() == 1{
                    if loss.eq(&Loss::MSE){
                        if input_checked.dims().len() == 2{
                            output_checked = output_checked.reshape((1,
                                output_checked.shape().dims().get(0).unwrap().to_owned() 
                                )
                            ).unwrap();
                        }
                        else if input_checked.dims().len() == 3{
                            output_checked = output_checked.reshape((1,
                                1,
                                output_checked.shape().dims().get(0).unwrap().to_owned() 
                                )
                            ).unwrap();
                        }
                        else if input_checked.dims().len() == 4{
                            output_checked = output_checked.reshape((1,
                                1,
                                1, 
                                output_checked.shape().dims().get(0).unwrap().to_owned(), 
                                )
                            ).unwrap();
                        }
                        else {
                            panic!("Not supported dimension");
                        }
                    }
                }
        
                if loss.ne(&Loss::MSE){
                    panic!("Not supported so far");
                }
                input_checked = input_checked.to_dtype(DType::F32).unwrap();
                output_checked = output_checked.to_dtype(DType::F32).unwrap();
        
                // Apply loss
                let lossed =  match loss {
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
        
                let enumvalue: (f64,u8) = match optimizer {
                    Optimizers::SGD(lrate) => (lrate.to_owned(), 1),
                    Optimizers::Adam(lrate) => (lrate.to_owned(),2),
                    Optimizers::None(_lrate) => (0.0,0),
                };
        
                // Apply optimizer 
                // Also see https://github.com/huggingface/candle/issues/1509#issuecomment-1872916766
                if enumvalue.1 == 1 {
                    let mut optimized: SGD = candle_nn::SGD::new(varmap.all_vars(), enumvalue.0).unwrap();
                    let _ = optimized.backward_step(&lossed_checked);
                    if bestloss.ge(&lossed_checked.to_vec0::<f32>().unwrap()){
                        if lossed_checked.to_vec0::<f32>().unwrap().ne(&0.0) {
                            bestloss = lossed_checked.to_vec0::<f32>().unwrap();
                            snapshot = Some(varmap.clone());
                        }
                    }
                }
                else if enumvalue.1 == 2 {
                    let adamw_params = candle_nn::ParamsAdamW {
                        lr: enumvalue.0,
                        ..Default::default()
                    };
                    let mut optimized: AdamW = candle_nn::AdamW::new(varmap.all_vars(), adamw_params).unwrap();
                    let _ = optimized.backward_step(&lossed_checked);
                    if bestloss.ge(&lossed_checked.to_vec0::<f32>().unwrap()){
                        if lossed_checked.to_vec0::<f32>().unwrap().ne(&0.0) {
                            bestloss = lossed_checked.to_vec0::<f32>().unwrap();
                            snapshot = Some(varmap.clone());
                        }
                    }
                }
            }
        }
        println!("Best loss {} ",bestloss);
        varmap.data().clone_from(&snapshot.clone().unwrap().data());
        return snapshot.unwrap();
    }

}
