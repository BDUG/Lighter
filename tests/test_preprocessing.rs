
#[allow(unused)]
use candlelighter::prelude::*;
use candlelighter::*;
use candlelighter::{preprocessing::featurescaling, recurrenttypes::RecurrentType};


#[test]
fn test_standardizednormalized()-> anyhow::Result<()> {

    let dev = candle_core::Device::cuda_if_available(0).unwrap();  
    let x_test: [[f32; 3]; 1] = [ [1., 2.5, 4.] ];
    let scaling = featurescaling::FeatureScaling::new(Tensor::new(&x_test, &dev).unwrap());
    let mut rst = scaling.min_max_normalization();
    let rst1 = scaling.z_score();
    
    //assert_eq!(rst, [0.0000, 0.5000, 1.0000]);
    //assert_eq!(rst1, [-1.2247, 0.0000, 1.2247]);
    //assert_eq!(scaling.min_max_normalization_reverse(rst.clone()), [1.0000, 2.5000, 4.0000]);
    //assert_eq!(scaling.z_score_reverse(rst1.clone()) , [1.0000, 2.5000, 4.0000]);

    anyhow::Ok(())
}
