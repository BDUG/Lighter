
#[allow(unused)]
use candlelighter::prelude::*;


#[test]
fn test_conv()-> anyhow::Result<()> {
    let varmap2 = VarMap::new();
    let dev = candle_core::Device::cuda_if_available(0).unwrap();  
    let z: [[[f32; 2]; 1]; 6] = [ [[1., 2.]] , [[2., 1.]] ,[[3., 4.]], [[5., 6.]], [[5., 5.]] , [[4., 5.]]];
    let mut layers2: Vec<Box<dyn Trainable>> = vec![];
    let mut name4 = String::new();
    name4.push_str("conv");
    layers2.push(Box::new(Conv::new(Tensor::new(&z, &dev).unwrap(), 1,2, 3, 4, 5, &dev, &varmap2, name4)));  
    
    let model2 = SequentialModel::new(varmap2, layers2);
    model2.save_model("model.model");
    model2.load_model("model.model",&dev);

    anyhow::Ok(())
}

#[test]
fn test_pooling()-> anyhow::Result<()> {
    let varmap2 = VarMap::new();
    let dev = candle_core::Device::cuda_if_available(0).unwrap();  
    
    let mut layers2: Vec<Box<dyn Trainable>> = vec![];
    let mut name4 = String::new();
    name4.push_str("pooling");
    layers2.push(Box::new(Pooling::new(PoolingType::MAX , 1,1 , &dev, &varmap2, name4)));  
    
    let model2 = SequentialModel::new(varmap2, layers2);
    model2.save_model("model.model");
    model2.load_model("model.model",&dev);

    anyhow::Ok(())
}