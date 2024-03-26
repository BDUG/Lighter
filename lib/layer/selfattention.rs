#[allow(unused)]
pub use crate::prelude::*;

pub struct SelfAttention {
    pub query_dim: usize,
    pub heads: usize,
    pub dim_head: usize,
    pub attention: Attention,
    pub key_value_mapping: candle_nn::Linear,
    pub kv: Tensor,
    pub device: Device,
    pub name: String,
}

pub trait SelfAttentionTrait {
    fn new(query_dim: usize, heads: usize, dim_head: usize, input_size: usize,device: &Device, varmap : &VarMap, name: String) -> Self;
}


impl SelfAttentionTrait for SelfAttention {
    fn new(query_dim: usize, heads: usize, dim_head: usize, input_size: usize,device: &Device, varmap : &VarMap, name: String) -> Self{
        let tmp_name = name.clone();
        let vs = VarBuilder::from_varmap(varmap, DType::F32, &device);
        let tmp_attention: Option<Attention> = Some(Attention::new(query_dim, heads, dim_head , false, vs).unwrap());
        let vs2 = VarBuilder::from_varmap(varmap, DType::F32, &device);
        let tmp_key_value_mapping = candle_nn::linear(input_size, query_dim, vs2.pp("kv_mapper.1")).unwrap();

        let batchsize = 1; // Due to outer loop
        let channel = 1; // FIX ME
        let height = input_size;//1;
        let weight = 1;
        // Tensor::randn(0f32, 1., (batchsize,channel,height,weight), &device).unwrap(),
        Self {
            query_dim: query_dim,
            heads: heads,
            dim_head: dim_head,
            attention : tmp_attention.unwrap(),
            key_value_mapping: tmp_key_value_mapping,
            kv : Tensor::ones((batchsize,channel,height,weight), DType::F32,&device).unwrap(),
            device : device.clone(),
            name: tmp_name.clone(),
        }
    }

}


impl Trainable for SelfAttention {
    
    fn forward( &self, input: Tensor) -> Tensor {
        //self.kv.silu().unwrap().apply(&self.key_value_mapping).unwrap();
        let batchsize = 1; // Due to outer loop
        let channel = 1; // FIX ME
        let height = input.shape().dims()[0];
        let weight = input.shape().dims()[1];
        return self.attention.forward(&input.reshape((batchsize,channel,height,weight)).unwrap().clone(), & self.kv.reshape((1,weight,1)).unwrap().clone()).unwrap().clone();
    }

    fn typ(&self) -> String {
        "SelfAttention".into()
    }

    fn input_perceptrons(&self) -> u32{
        return 1.0 as u32;
    }
    fn output_perceptrons(&self) -> u32{
        return 1.0 as u32;
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
    
}
