
#[allow(unused)]
pub use crate::prelude::*;

use crate::topk::{TopK, TopKTrait};


pub struct SparseMoE {
    pub experts: Vec<SequentialModel>,
    pub input_dim: usize, 
    pub output_dim: usize, 
    pub gate: SequentialModel,
    pub device: Device,
    pub name: String,
}


pub trait SparseMoETrait {
    fn create_expert(input_dim: usize, output_dim: usize, name: String, dev: &Device) -> SequentialModel;
    fn create_gate(input_dim: usize,  dev: &Device) -> SequentialModel;
    fn new(num_of_experts: usize, input_dim: usize, output_dim: usize, device: &Device, varmap : &VarMap, name: String) -> Self;
}


impl SparseMoETrait for SparseMoE {

    fn create_expert(input_dim: usize, output_dim: usize, name: String, dev: &Device) -> SequentialModel {     
        let mut layers: Vec<Box<dyn Trainable > > = Vec::new();

        let varmap = VarMap::new();
        let mut name1 = String::new();
        name1.push_str("fc1_");
        name1.push_str(&name);
        layers.push(Box::new(Dense::new(input_dim/2, input_dim, Activations::Relu, &dev, &varmap, name1 )));
        let mut name2 = String::new();
        name2.push_str("fc2_");
        name2.push_str(&name);
        layers.push(Box::new(Dense::new(input_dim/4, input_dim/2, Activations::Relu, &dev, &varmap, name2 )));
        let mut name3 = String::new();
        name3.push_str("fc3_");
        name3.push_str(&name);
        layers.push(Box::new(Dense::new(output_dim, input_dim/4, Activations::Relu, &dev, &varmap, name3 )));

        return SequentialModel::new(varmap, layers);
    }

    fn create_gate(input_dim: usize, dev: &Device) -> SequentialModel {
        let mut layers: Vec<Box<dyn Trainable>> = Vec::new();
        let varmap = VarMap::new();

        let mut name1 = String::new();
        name1.push_str("fc1_gate");
        layers.push(Box::new(Dense::new(input_dim/2, input_dim, Activations::Relu, &dev, &varmap, name1 )));
        let mut name2 = String::new();
        name2.push_str("fc2_gate");

        layers.push(Box::new(Dense::new(input_dim/2, input_dim/2, Activations::Relu, &dev, &varmap, name2 )));

        SequentialModel::new(varmap, layers)
    }


    fn new(num_of_experts: usize, input_dim: usize, output_dim: usize, device: &Device, varmap : &VarMap, name: String) -> Self{
        let tmp_name = name.clone();

        let mut name1 = String::new();
        name1.push_str(&tmp_name);
        
        let _vs = VarBuilder::from_varmap(varmap, DType::F32, &device);

        let mut tmp_experts : Vec<SequentialModel> = Vec::new();
        for _i in 0..num_of_experts {
            let mut _name2: String = name1.clone();
            _name2.push_str(&_i.to_string());
            tmp_experts.push(SparseMoE::create_expert(input_dim, output_dim, _name2, device));
        }

        Self {
            experts : tmp_experts,
            input_dim: input_dim,
            output_dim: output_dim,
            gate: SparseMoE::create_gate(input_dim, device),
            device : device.clone(),
            name: tmp_name.clone(),
        }
    }
    

}


impl Trainable for SparseMoE {
    
    fn forward( &self, input: Tensor) -> Tensor {
        let mut _expert_outputs: Vec<Tensor> = Vec::new();
        let mut _cloned_input= input.clone();
        for _expert in &self.experts {
            _expert_outputs.push(_expert.forward(_cloned_input.clone()));
        }
        let gateoutput = self.gate.forward(_cloned_input);
        let topk = TopK::new();
        let mut _gate_outputs = gateoutput.flatten_all().unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();
        let _positions = topk.top_k_positions(_gate_outputs.clone(), 2);

        let mut _expert_outputs_new: Vec<Tensor> = Vec::new();
        for _expert_position in _positions {
            let _votingweight_given_to_expert = _gate_outputs.get(_expert_position).unwrap();
            let _expert_opinion = _expert_outputs.get(_expert_position).unwrap();
            let _expert_opinion_vector = _expert_opinion.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let _expert_opinion_result_vetor: Vec<f32> = _expert_opinion_vector.iter().map(|x| x * _votingweight_given_to_expert).collect();

            let new_output = Tensor::new(_expert_opinion_result_vetor,input.device()).unwrap();
            _expert_outputs_new.push(new_output);
        }
        let _first_expert_result_weighted = _expert_outputs_new.pop().unwrap();
        let mut _expert_result_weighted_sum = _first_expert_result_weighted.clone();
        for _expert_result_weighted_element in _expert_outputs_new {
            _expert_result_weighted_sum = _expert_result_weighted_sum.add(&_expert_result_weighted_element).unwrap();
        }
        // FIX ME: Not generic
        return _expert_result_weighted_sum.clone().reshape(input.shape()).unwrap();
    }

    fn typ(&self) -> String {
        "SparseMoE".into()
    }

    fn input_perceptrons(&self) -> u32{
        return 1.0 as u32;
    }
    fn output_perceptrons(&self) -> u32{
        return 1.0 as u32;
    }

    fn as_any(&self) -> &(dyn Any + 'static) {
        todo!();
        //self
    }
    
}
