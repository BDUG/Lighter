#[allow(unused)]
use crate::prelude::*;
use crate::parallelmodeltypes::ParallelModelType;


// ###########################################################################
// Parallel model 
// 

pub struct ParallelModel {
    pub layers: Vec<Box<dyn Trainable>> ,
    pub types: ParallelModelType,
    pub optimizer: Optimizers,
    pub loss: Loss,
    pub gate: SequentialModel,
    pub varmap: VarMap
}

impl Fitable for ParallelModel {
}

impl Predictable for ParallelModel  {
}

impl ModelSerialization for ParallelModel  {
}

impl DoPrediction for ParallelModel {
    fn predict(&self, x: &Tensor) -> Option<Vec<Tensor>> {
        if self.types == ParallelModelType::Split {
            if let Some(value) = self.predicting(&self.layers, x) {
                return Some(value);
            }
        }
        let mut _expert_outputs_new = Vec::new();
        for layer in &self.layers {
            let output = layer.forward(x.clone());
            _expert_outputs_new.push(output);
        }
        let mut result = Vec::new();
        result.push(self.calculate_weighted_sum(x.clone(), &_expert_outputs_new).clone());
        return Some(result);
    }
}

impl ParallelModel {
    pub fn new(types: ParallelModelType, device: &Device, varmap:  VarMap, layers: Vec<Box<dyn Trainable>>)  -> Self {
        let mut _len = 1;
        
        if types == ParallelModelType::Split {
            _len = layers.len();
        }
        else if types == ParallelModelType::Merge {
            _len = 1;
        }
        else {
            panic!("Unknown type");
        }

        Self {
            layers: layers,
            types: types,
            optimizer: Optimizers::None(0.0),
            loss: Loss::None,
            gate: ParallelModel::create_gate(_len, "gate".to_string(), &device),
            varmap: varmap,
        }
    }

    pub fn summary(&self) {
        let mut total_param = 0;
        let mut res = "\nModel Parallel\n".to_string();
        res.push_str("-------------------------------------------------------------\n");
        res.push_str("Layer (Type)\t\t Output shape\t\t No.of params\n");
        for layer in self.layers.iter() {
            let a = layer.input_perceptrons();
            let b = layer.output_perceptrons();
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
        if self.types == ParallelModelType::Split {  
            self.varmap = self.fitting(&self.layers, &self.loss, &self.optimizer, &self.varmap, epochs, _verbose, &x, &_y);
        }
        else {
            //for layer in &mut self.layers {
            //    layer.forward(x.clone());
            //}
            let mut _expert_outputs_new: Vec<Tensor> = Vec::new();
            for _layer in self.layers.iter() {
                let new_output = _layer.forward(x.clone());
                _expert_outputs_new.push(new_output);
            }
            self.calculate_weighted_sum(x.clone(), &_expert_outputs_new);
        }
    }

    /*
    pub fn predict(&self, x: Tensor) -> Vec<Tensor> {
        if self.types == ParallelModelType::Split {
            if let Some(value) = self.predicting(&self.layers, &x) {
                return value;
            }
        }
        let mut _expert_outputs_new = Vec::new();
        for layer in &self.layers {
            let output = layer.forward(x.clone());
            _expert_outputs_new.push(output);
        }
        let mut result = Vec::new();
        result.push(self.calculate_weighted_sum(x.clone(), &_expert_outputs_new).clone());
        return result;
    }
    */

    fn calculate_weighted_sum(&self, x: Tensor, expert_outputs: &Vec<Tensor>) -> Tensor{
        let gateoutput = self.gate.forward(x.clone());
        let mut _gate_outputs = gateoutput.flatten_all().unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();

        let mut _expert_result_new: Vec<Tensor> = Vec::new();
        for (position, _expert_opinion) in expert_outputs.iter().enumerate() {
            let _votingweight_given_to_expert = _gate_outputs.get(position).unwrap();

            let _expert_opinion_vector = _expert_opinion.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let _expert_opinion_result_vetor: Vec<f32> = _expert_opinion_vector.iter().map(|somex| somex * _votingweight_given_to_expert).collect();

            let new_output = Tensor::new(_expert_opinion_result_vetor,x.clone().device()).unwrap();
            _expert_result_new.push(new_output);
        }
        let _first_expert_result_weighted = _expert_result_new.pop().unwrap();
        let mut _expert_result_weighted_sum = _first_expert_result_weighted.clone();
        for _expert_result_weighted_element in _expert_result_new {
            _expert_result_weighted_sum = _expert_result_weighted_sum.add(&_expert_result_weighted_element).unwrap();
        }
        return _expert_result_weighted_sum.clone().reshape(x.clone().shape()).unwrap();
    }


    fn create_gate(input_dim: usize,  name: String, dev: &Device) -> SequentialModel {
        let mut layers: Vec<Box<dyn Trainable>> = vec![];
        let varmap = VarMap::new();

        let mut name1 = String::new();
        name1.push_str("fc1_gate");
        name1.push_str(&name);
        layers.push(Box::new(Dense::new(input_dim/2, input_dim, Activations::Relu, &dev, &varmap, name1 )));
        let mut name2 = String::new();
        name2.push_str("fc2_gate");
        name2.push_str(&name);

        let mut _num_perceptrons: i32 = 1;
        if input_dim >= 2 {
            _num_perceptrons = (input_dim as f32/2.0).round() as i32;
        }

        layers.push(Box::new(Dense::new(_num_perceptrons as usize, _num_perceptrons as usize, Activations::Relu, &dev, &varmap, name2 )));

        return SequentialModel::new(varmap, layers);
    }

    /*
    pub fn load_weights(&mut self, path: &str, device: &Device) {
        // FIX ME
        //self.varmap = self.load_weights_generic(path, device);
        self.load_weights_generic(path, &self.varmap, device);
    }
    */

    pub fn load_model(&self, path: &str, device : &Device) -> ParallelModel {
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

            self.load_layer(new_type, elem, device, &mut layers, &self.varmap);
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

        let mut new_model= ParallelModel::new(self.types.clone(), device, varmap,layers);
        new_model.compile(new_optimizer, Loss::from_string(new_loss));
        return new_model;
    }
  
}


impl Trainable for ParallelModel {
    fn forward(&self, input: Tensor) -> Tensor {
        let mut input_checked = input.clone();
        for layer in self.layers.iter() {
            input_checked = layer.forward(input_checked).clone();
        }

        return input_checked;
    }

    fn typ(&self) -> String {
        "Parallel".into()
    }

    fn input_perceptrons(&self) -> u32{
        return 1.0 as u32;
    }
    fn output_perceptrons(&self) -> u32{
        return 1.0 as u32;
    }

    fn as_any(&self) -> &dyn Any {
        return self;
    }
    
}

impl Serialize for ParallelModel {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<<S as Serializer>::Ok, <S as Serializer>::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("parallel", 3)?;
        let _ = state.serialize_field("types", &self.types);
        let _ = state.serialize_field("optimizer", &self.optimizer);
        let _ = state.serialize_field("loss", &self.loss);      
        let _ = state.serialize_field("layers", &self.layers);
        return Ok(state.end().unwrap());
    }
}

