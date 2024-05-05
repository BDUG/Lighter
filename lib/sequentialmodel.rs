

#[allow(unused)]
use crate::prelude::*;

// ###########################################################################
// Sequential model 
// 


pub struct SequentialModel {
    pub layers: Vec<Box<dyn Trainable>> ,
    pub optimizer: Optimizers,
    pub loss: Loss,
    pub varmap: VarMap
}

impl Fitable for SequentialModel { 
}

impl Predictable for SequentialModel {
}

impl ModelSerialization for SequentialModel {
}


impl SequentialModel {
    pub fn new(varmap: VarMap, layers: Vec<Box<dyn Trainable>>) -> Self {
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
        self.varmap = self.fitting(&self.layers, &self.loss, &self.optimizer, &self.varmap, epochs, _verbose, &x, &_y);
    }

    pub fn predict(&self, x: Tensor) -> Vec<Tensor> {    
        if let Some(value) = self.predicting(&self.layers, &x) {
            return value;
        }
        return Vec::new();
    }

    /*
    pub fn save_weights(&self, path: &str) {
        self.save_weights_generic(&self.varmap, path);
    }

    pub fn load_weights(&mut self, path: &str, device: &Device) {
        //self.varmap = self.load_weights_generic(path, device);
        self.load_weights_generic(path, &self.varmap, device);
    }
    */

    pub fn save_model(&self, path: &str) {
        let mut file = File::create(path).unwrap();
        let j = serde_json::to_string(&self).unwrap();
        file.write(&j.as_bytes()).unwrap();
    }


    pub fn load_model(&self, path: &str, device : &Device) -> SequentialModel {
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

        let mut new_model= SequentialModel::new(varmap,layers);
        new_model.compile(new_optimizer, Loss::from_string(new_loss));
        return new_model;
    }

  
  

}

impl Trainable for SequentialModel {
    fn forward(&self, input: Tensor) -> Tensor {
        let mut input_checked = input.clone();
        for layer in self.layers.iter() {
            //println!("{} ",layer.typ());
            input_checked = layer.forward(input_checked).clone();
        }

        return input_checked;
    }

    fn typ(&self) -> String {
        "Sequential".into()
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


impl Serialize for SequentialModel {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<<S as Serializer>::Ok, <S as Serializer>::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("sequential", 3)?;
        let _ = state.serialize_field("optimizer", &self.optimizer);
        let _ = state.serialize_field("loss", &self.loss);      
        let _ = state.serialize_field("layers", &self.layers);
        return Ok(state.end().unwrap());
    }
}
