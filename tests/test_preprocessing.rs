
#[allow(unused)]
use candlelighter::prelude::*;
use candlelighter::preprocessing::featurescaling;


#[test]
fn test_standardizednormalized()-> anyhow::Result<()> {

    let dev = candle_core::Device::cuda_if_available(0).unwrap();  
    let x_test: [[f32; 3]; 1] = [ [1., 2.5, 4.] ];
    let scaling = featurescaling::FeatureScaling::new(Tensor::new(&x_test, &dev).unwrap());
    scaling.min_max_normalization();
    scaling.z_score();

    anyhow::Ok(())
}
