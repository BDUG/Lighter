

#[allow(unused)]
pub use crate::prelude::*;

pub struct FeatureScaling {
    input : Tensor
}

pub struct MinMaxAttributes {
    min : f32,
    max : f32
}

pub struct ZScoreAttributes {
    mean : f32,
    std_deviation : f32
}

pub trait FeatureScalingTrait {
    fn new(input: Tensor) -> Self;
    fn min_max_normalization_attributes(&self) -> MinMaxAttributes;
    fn min_max_normalization(&self) -> Tensor;
    fn min_max_normalization_other(&self, normalized: Tensor) -> Tensor;
    fn min_max_normalization_reverse(&self, normalized: Tensor) -> Tensor;
    fn z_score_attributes(&self) -> ZScoreAttributes;
    fn z_score(&self) -> Tensor;
    fn z_score_other(&self, standardized: Tensor) -> Tensor;
    fn z_score_reverse(&self, standardized: Tensor) -> Tensor;
}

impl FeatureScalingTrait for FeatureScaling {
    fn new(input: Tensor) -> Self {
        Self {
            input: input
        }
    }

    fn min_max_normalization_attributes(&self) -> MinMaxAttributes
    {
        let tmp_tensor = self.input.clone();
        let number_max: f32 = tmp_tensor.flatten_all().unwrap().max(0).unwrap().to_vec0().unwrap();
        let number_min: f32 = tmp_tensor.flatten_all().unwrap().min(0).unwrap().to_vec0().unwrap();
        let result = MinMaxAttributes{
            max: number_max,
            min: number_min
        };
        return result;
    }

    fn z_score_attributes(&self) -> ZScoreAttributes
    {
        let tmp_tensor = self.input.clone();
        let number_mean: f32 = tmp_tensor.flatten_all().unwrap().mean(0).unwrap().to_vec0().unwrap();
        
        let variance = tmp_tensor.flatten_all().unwrap().
        to_vec1::<f32>().unwrap().iter().map(|&x| (x - number_mean).powi(2)).sum::<f32>() as f64/ tmp_tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap().len() as f64;
        let number_std_deviation = variance.sqrt() as f32;
        let result = ZScoreAttributes{
            mean: number_mean,
            std_deviation: number_std_deviation
        };
        return result;
    }

    /**
     * Normalization is a scaling technique in which values are shifted and rescaled so that they
     * end up ranging between 0 and 1.
     *
     * - Useful when the distribution of the data is unknown or not Gaussian
     * - Sensitive to outliers
     * - Retains the shape of the original distribution
     * - May not preserve the relationships between the data points
     */
    fn min_max_normalization(&self) -> Tensor {
        let tmp_tensor = self.input.clone();

        let attributes = self.min_max_normalization_attributes();

        let normalized_numbers: Vec<f32> = tmp_tensor.flatten_all().unwrap().
        to_vec1::<f32>().unwrap().iter().map(|x| (*x - attributes.min) / (attributes.max - attributes.min)).collect();
    
        return Tensor::new(normalized_numbers, self.input.device()).unwrap().reshape(self.input.shape()).unwrap();
    }


    fn min_max_normalization_other(&self, normalized: Tensor) -> Tensor {

        let attributes = self.min_max_normalization_attributes();

        let normalized_numbers: Vec<f32> = normalized.flatten_all().unwrap().
        to_vec1::<f32>().unwrap().iter().map(|x| (*x - attributes.min) / (attributes.max - attributes.min)).collect();
    
        return Tensor::new(normalized_numbers, normalized.device()).unwrap().reshape(normalized.shape()).unwrap();
    }

    fn min_max_normalization_reverse(&self, normalized: Tensor) -> Tensor {
        let attributes = self.min_max_normalization_attributes();

        let denormalized_numbers: Vec<f32> = normalized.flatten_all().unwrap().
        to_vec1::<f32>().unwrap().iter().map(|x| (x * ( attributes.max - attributes.min ) + attributes.min) ).collect();
    

        return Tensor::new(denormalized_numbers, self.input.device()).unwrap();
    }

    /**
     * Data involves scaling data values so that they have a mean of zero and standard deviation of one.
     *
     * - Useful when the distribution of the data is Gaussian or unknown
     * - Less sensitive to outliers
     * - Changes the shape of the original distribution
     * - Preserves the relationships between the data points
     */
    fn z_score(&self) -> Tensor {
        let tmp_tensor = self.input.clone();

        let attributes = self.z_score_attributes();

        let normalized_numbers: Vec<f32> = tmp_tensor.flatten_all().unwrap().
        to_vec1::<f32>().unwrap().iter().map(|x| (*x - attributes.mean) / (attributes.std_deviation)).collect();
    
        return Tensor::new(normalized_numbers, self.input.device()).unwrap().reshape(self.input.shape()).unwrap();
    }

    fn z_score_other(&self, standardized: Tensor) -> Tensor {

        let attributes = self.z_score_attributes();

        let normalized_numbers: Vec<f32> = standardized.flatten_all().unwrap().
        to_vec1::<f32>().unwrap().iter().map(|x| (*x - attributes.mean) / (attributes.std_deviation)).collect();
    
        return Tensor::new(normalized_numbers, standardized.device()).unwrap().reshape(standardized.shape()).unwrap();
    }

    fn z_score_reverse(&self, standardized: Tensor) -> Tensor {
        let attributes = self.z_score_attributes();

        let normalized_numbers: Vec<f32> = standardized.flatten_all().unwrap().
        to_vec1::<f32>().unwrap().iter().map(|x| (*x * attributes.std_deviation) + attributes.mean ).collect();
    
        return Tensor::new(normalized_numbers, self.input.device()).unwrap();
    }

}