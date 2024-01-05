

#[allow(unused)]
use crate::prelude::*;

pub struct Sequential {
    pub layers: Vec<Box<dyn Trainable>> ,
    pub optimizer: Optimizers,
    pub loss: Loss,
    pub varmap: VarMap,
}
impl Sequential {
    pub fn new(varmap: VarMap,layers: Vec<Box<dyn Trainable>>) -> Self {
        Self {
            layers: layers,
            optimizer: Optimizers::None(0.0),
            loss: Loss::None,
            varmap: varmap,
        }
    }

    pub fn summary(&self) {
        let mut total_param = 0;
        let mut res = "\nModel Sequential\n".to_string();
        res.push_str("-------------------------------------------------------------\n");
        res.push_str("Layer (Type)\t\t Output shape\t\t No.of params\n");
        for layer in self.layers.iter() {
            let a = layer.inputPerceptrons();
            let b = layer.outputPerceptrons();
            total_param += a + b;
            res.push_str(&format!("{}\t\t\t  (None, {})\t\t  {}\n", layer.typ(), b, a + b));
        }
        res.push_str("-------------------------------------------------------------\n");
        res.push_str(&format!("Total params: {}\n", total_param));
        println!("{}", res);
    }

    pub fn compile(&mut self, optimizer: Optimizers, loss: Loss) {
        self.optimizer = optimizer;
        self.loss = loss;
    }

    pub fn fit(&mut self, x: Tensor, _y: Tensor, epochs: usize ,_verbose: bool) {
        for _i in 0..epochs {
            if _verbose{
                println!("Epoche {}",_i);
            }
            
            for elementnumber in 0.. x.dims().len() {
                let mut input_checked = match x.get(elementnumber) {
                    Ok(element) => element,
                    Err(error) => panic!("{}",error.to_string()),
                };

                let output_checked = match _y.get(elementnumber) {
                    Ok(element) => element,
                    Err(error) => panic!("{}",error.to_string()),
                };
                
                //let mut input_stack = vec![];
                for layer in self.layers.iter() {
                    input_checked = layer.forward(input_checked).clone();
                }
                // Apply loss
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

                let enumvalue: (f64,u8) = match self.optimizer {
                    Optimizers::SGD(lrate) => (lrate, 1),
                    Optimizers::Adam(lrate) => (lrate,2),
                    Optimizers::None(lrate) => (0.0,0),
                };
                // Apply optimizer 
                // Also see https://github.com/huggingface/candle/issues/1509#issuecomment-1872916766
                if enumvalue.1 == 1 {
                    let mut optimized: SGD = candle_nn::SGD::new(self.varmap.all_vars(), enumvalue.0).unwrap();
                    optimized.backward_step(&lossed_checked);
                }
                else if enumvalue.1 == 2 {
                    let adamw_params = candle_nn::ParamsAdamW {
                        lr: enumvalue.0,
                        ..Default::default()
                    };
                    let mut optimized: AdamW = candle_nn::AdamW::new(self.varmap.all_vars(), adamw_params).unwrap();
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

    pub fn save_weights(&self, path: &str) {
        let mut toserialize : Vec<SerializedTensor> = Vec::new();

        let tmp = self.varmap.data();
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

    pub fn load_weights(&self, path: &str, device: &Device) {
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
                Some(x) => x.clone(),
                None    => panic!("Unknown state"),
            };
            /* Construct via tensor
            let dimension = match _dimension {
                Some(x) => x.clone().iter().map(|x| x.as_u64().unwrap() as usize).collect::<Vec<usize>>(),
                None    => panic!("Unknown state"),
            };
            let mut resulttensor = Tensor::new(values, &device).unwrap();
            if dimension.len() > 1{        
                resulttensor = resulttensor.clone().reshape(dimension.as_slice()).unwrap();
            }
            */
            self.varmap.data().lock().unwrap().insert(name.to_string(),Var::new(values, &device).unwrap());
        }
    }

    pub fn save_model(&self, path: &str) {
        let mut file = File::create(path).unwrap();
        let j = serde_json::to_string(&self).unwrap();
        file.write(&j.as_bytes()).unwrap();
    }


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
        let rst = match value {
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

    pub fn load_model(&self, path: &str, device : &Device) -> Sequential {
        let value = fs::read_to_string(path).unwrap();
        let json: serde_json::Value =
            serde_json::from_str(&value).unwrap();
        
        let _value_map = match json {
            Value::Object(ref map) => map,
            _ => panic!("Unknown state"),
        };
        let numberoflayers_string = _value_map.get_key_value("layers");
        let rst = match numberoflayers_string {
            Some(x) => x.1,
            None    => panic!("Unknown state"),
        };
        let varmap = VarMap::new();
        let mut layers: Vec<Box<dyn Trainable>> = vec![];
        let layerslist = rst.as_array().unwrap();
        for i in 0..layerslist.len(){
            let elem = layerslist.get(i).unwrap().as_object().unwrap();

            let new_type = self.extractjson_value_str(elem, "type");
            if new_type.eq("Dense"){
                let new_perceptrons = self.extractjson_value_u64(elem, "perceptrons");
                let new_previousperceptrons = self.extractjson_value_u64(elem, "previousperceptrons");
                let new_name = self.extractjson_value_str(elem, "name");
                let tmp_activation = self.extractjson_value_str(elem, "activation");
                let new_activation = Activations::from_string(tmp_activation.to_string());
                
                let tmp = Box::new(Dense::new(new_perceptrons as usize, new_previousperceptrons as usize, new_activation, device, &self.varmap, new_name));
                layers.push(tmp);
            }
            else if new_type.eq("Pooling"){
                let new_poolingtype = self.extractjson_value_str(elem, "poolingtype");
                let new_kernelsize = self.extractjson_value_u64(elem, "kernelsize") as usize;
                let new_stride = self.extractjson_value_u64(elem, "stride") as usize;
                let new_name = self.extractjson_value_str(elem, "name");
        
                let tmp = Box::new(Pooling::new(PoolingType::from_string(new_poolingtype), new_kernelsize, new_stride as usize, &device, &self.varmap, new_name));
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

                let tmp = Box::new(Conv::new(new_tensor, new_dimensionality,new_padding, new_stride, new_dilation, new_groups,device,&self.varmap,new_name));
                layers.push(tmp);
            }
            else{
                panic!("Unknown layer type")
            }
        }
        let tmp_optimizer = _value_map.get_key_value("optimizer").unwrap().1;
        let new_optimizer: Optimizers;
        if tmp_optimizer.is_object(){
            let tmp_optimizer_obj = tmp_optimizer.as_object().unwrap();
            if !tmp_optimizer_obj.get_key_value("SGD").is_none() {
                let learning_rate = self.extractjson_value_f64(tmp_optimizer_obj, "SGD");
                new_optimizer = Optimizers::SGD(learning_rate);
            }      
            else if !tmp_optimizer_obj.get_key_value("Adam").is_none() {
                let learning_rate = self.extractjson_value_f64(tmp_optimizer_obj, "Adam");
                new_optimizer = Optimizers::Adam(learning_rate);
            }         
            else if !tmp_optimizer_obj.get_key_value("None").is_none() {
                let learning_rate = self.extractjson_value_f64(tmp_optimizer_obj, "None");
                new_optimizer = Optimizers::None(learning_rate);
            }   
            else {
                panic!("undefined optimizer");
            } 
        }
        else {
            new_optimizer = Optimizers::from_string(self.extractjson_value_str(_value_map, "optimizer"));
        }
        
        let new_loss = self.extractjson_value_str(_value_map, "loss");

        let mut new_model= Sequential::new(varmap,layers);
        new_model.compile(new_optimizer, Loss::from_string(new_loss));
        return new_model;
    }
  
    pub fn buildname(&self, name: &str,  number: i32) -> String {
        let mut result = String::from(name);
        result.push_str(&number.to_string());
        return result;
    }
  

}



impl Serialize for Sequential {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<<S as Serializer>::Ok, <S as Serializer>::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("sequential", 3)?;
        state.serialize_field("optimizer", &self.optimizer);
        state.serialize_field("loss", &self.loss);      
        state.serialize_field("layers", &self.layers);
        return Ok(state.end().unwrap());
    }
}
