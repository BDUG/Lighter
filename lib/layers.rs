#[allow(unused)]
use crate::prelude::*;

pub trait Trainable {
    fn forward(&self, input: Tensor) -> Tensor;
    fn typ(&self) -> String;
    fn input_perceptrons(&self) -> u32;
    fn output_perceptrons(&self) -> u32;
    fn as_any(&self) -> &dyn Any;
}

impl Serialize for dyn Trainable {

    fn serialize<S>(&self, serializer: S) -> std::result::Result<<S as Serializer>::Ok, <S as Serializer>::Error>
    where
        S: Serializer,
    {
        // FIXME Push logic to layer itself
        if self.typ().eq("Dense") {
            let mut state = serializer.serialize_struct("Dense", 4)?;
            // One of two ways to downcast in Rust
            let dense: &Dense = match self.as_any().downcast_ref::<Dense>() {
                Some(b) => b,
                None => panic!("Not a Dense type"),
            };
            state.serialize_field("type", "Dense")?;
            state.serialize_field("perceptrons", &dense.perceptrons)?;
            state.serialize_field("previousperceptrons", &dense.previousperceptrons)?;
            state.serialize_field("activation", &dense.activation)?;
            state.serialize_field("name", &dense.name)?;
            return Ok(state.end().unwrap());
        }
        else if self.typ().eq("Normalization") {
            let mut state = serializer.serialize_struct("Pooling", 5)?;
            // One of two ways to downcast in Rust
            let normalization: &Normalization = match self.as_any().downcast_ref::<Normalization>() {
                Some(b) => b,
                None => panic!("Not a Normalization type"),
            };
            state.serialize_field("type", "Normalization")?;
            state.serialize_field("poolingtype", &normalization.axis)?;
            state.serialize_field("name", &normalization.name)?;
            return Ok(state.end().unwrap());
        }
        else if self.typ().eq("Pooling") {
            let mut state = serializer.serialize_struct("Pooling", 5)?;
            // One of two ways to downcast in Rust
            let dense: &Pooling = match self.as_any().downcast_ref::<Pooling>() {
                Some(b) => b,
                None => panic!("Not a Pooling type"),
            };
            state.serialize_field("type", "Pooling")?;
            state.serialize_field("poolingtype", &dense.poolingtype)?;
            state.serialize_field("kernelsize", &dense.kernelsize)?;
            state.serialize_field("stride", &dense.stride)?;
            state.serialize_field("name", &dense.name)?;
            return Ok(state.end().unwrap());
        }
        else if self.typ().eq("Conv") {
            let mut state = serializer.serialize_struct("Conv", 5)?;
            // One of two ways to downcast in Rust
            let conv: &Conv = match self.as_any().downcast_ref::<Conv>() {
                Some(b) => b,
                None => panic!("Not a Dense type"),
            };

            state.serialize_field("type", "Conv")?;
            let raw_tensor = &conv.kernel;
            let tensor = raw_tensor.flatten_all().unwrap();
            let ser_tensor = SerializedTensor {
                name: "undefined".to_owned(),
                dimension : raw_tensor.shape().dims().to_vec(),
                values : tensor.to_vec1::<f32>().unwrap(),
            };

            state.serialize_field("kernel", &ser_tensor)?;
            state.serialize_field("dimensionality", &conv.dimensionality)?;
            state.serialize_field("padding", &conv.padding)?;
            state.serialize_field("stride", &conv.stride)?;
            state.serialize_field("dilation", &conv.dilation)?;
            state.serialize_field("groups", &conv.groups)?;
            state.serialize_field("name", &conv.name)?;
            return Ok(state.end().unwrap());
        }
        panic!("Unknown layer type")
    }
}
