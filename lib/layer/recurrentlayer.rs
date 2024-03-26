
#[allow(unused)]
pub use crate::prelude::*;
use crate::recurrenttypes::RecurrentType;

pub struct Recurrent {
    pub recurrenttype: RecurrentType,
    pub indimension: usize,
    pub hiddendimension: usize,
    pub lstm: Option<LSTM>,
    pub gru: Option<GRU>,
    pub device: Device,
    pub name: String,
}

pub trait RecurrentLayerTrait {
    fn new(recurrenttype: RecurrentType, indimension: usize, hiddendimension: usize, device: &Device, varmap : &VarMap, name: String) -> Self;
}


impl RecurrentLayerTrait for Recurrent {
    fn new(recurrenttype: RecurrentType, indimension: usize, hiddendimension: usize, device: &Device, varmap : &VarMap, name: String) -> Self{
        let tmp_name = name.clone();
        let mut tmp_lstm = None;
        let mut tmp_gru = None;

         // Lazy Init
        if recurrenttype.eq(&RecurrentType::LSTM) {
            let vs = VarBuilder::from_varmap(varmap, DType::F32, &device);
            tmp_lstm = Some(lstm(indimension, hiddendimension, Default::default(), vs).unwrap());
        }
        if recurrenttype.eq(&RecurrentType::GRU) {
            let vs = VarBuilder::from_varmap(varmap, DType::F32, &device);
            tmp_gru = Some(gru(indimension, hiddendimension, Default::default(), vs).unwrap());
        }

        Self {
            recurrenttype: recurrenttype,
            indimension: indimension,
            hiddendimension: hiddendimension,
            lstm : tmp_lstm,
            gru : tmp_gru,
            device : device.clone(),
            name: tmp_name.clone(),
        }
    }

}



impl Trainable for Recurrent {
    
    fn forward(&self, input: Tensor) -> Tensor {
        let tmp = input.clone();
        if self.recurrenttype.eq(&RecurrentType::LSTM) {
            let state1 =  self.lstm.to_owned().unwrap().zero_state(1).unwrap();
            return self.lstm.to_owned().unwrap().step(&tmp, &state1).unwrap().c().clone();
        }
        else if self.recurrenttype.eq(&RecurrentType::GRU)  {
            let state2 =  self.gru.to_owned().unwrap().zero_state(1).unwrap();
            return self.gru.to_owned().unwrap().step(&tmp, &state2).unwrap().h().clone();
        }
        panic!("Unknown recurrent type")
    }

    fn typ(&self) -> String {
        "Recurrent".into()
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
