
use std::ops::Not;

#[allow(unused)]
pub use crate::prelude::*;
use crate::embeddingtypes::EmbeddingType;

pub struct Embed {
    pub embeddingtype: EmbeddingType,
    pub inputdimension: usize,
    pub hiddendimension: usize,
    pub embedding: Option<Embedding>,
    pub device: Device,
    pub name: String,
}

pub trait EmbeddingLayerTrait {
    fn new(embeddingtype: EmbeddingType,inputdimension: usize, hiddendimension: usize, device: &Device, varmap : &VarMap, name: String) -> Self;
}


impl EmbeddingLayerTrait for Embed {
    fn new(embeddingtype: EmbeddingType, inputdimension: usize,  hiddendimension: usize, device: &Device, varmap : &VarMap, name: String) -> Self{
        let tmp_name = name.clone();
        let mut tmp_embedding = None;

         // Lazy Init
        let vs = VarBuilder::from_varmap(varmap, DType::F32, &device);
        tmp_embedding = Some(embedding::embedding(inputdimension, hiddendimension, vs).unwrap());
        
        //

        Self {
            embeddingtype: embeddingtype,
            inputdimension: inputdimension,
            hiddendimension: hiddendimension,
            embedding : tmp_embedding,
            device : device.clone(),
            name: tmp_name.clone(),
        }
    }

}



impl Trainable for Embed {
    
    fn forward(&self, input: Tensor) -> Tensor {
        let new_tensor = input.to_dtype(DType::U32).unwrap();
        return self.embedding.as_ref().unwrap().forward(&new_tensor).unwrap().clone();
    }

    fn typ(&self) -> String {
        "Embedding".into()
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
