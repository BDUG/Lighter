use std::collections::HashMap;

#[allow(unused)]
use crate::prelude::*;
use hf_hub::api::sync::ApiBuilder;
use tokenizers::Tokenizer;


pub trait TransformerTrait {


    fn get_tokenizer(&self, parameter: &HashMap<String, Value>) -> Tokenizer{
        let mut api = ApiBuilder::new().with_progress(false);

        if parameter.contains_key("hf_tokenizer.json"){
            let tmp = parameter.get("hf_tokenizer.json").unwrap().as_str().unwrap();
            let tokenizer = Tokenizer::from_file(tmp).unwrap();
            return tokenizer;
        }
        else{
            if parameter.contains_key("hf_token"){
                api = api.with_token(Some(parameter.get("hf_token").unwrap().as_str().unwrap().into()));
            }
            if parameter.contains_key("hf_cachedir"){
                api = api.with_cache_dir(parameter.get("hf_cachedir").unwrap().as_str().unwrap().into());
            }
            else{
                api = api.with_cache_dir("./tmp".into());
            }

            if !parameter.contains_key("hf_model"){
                panic!("Unknown hf_model");
            }
            let model_id = Some(parameter.get("hf_model").unwrap().as_str().unwrap()).unwrap();

            let repo = api.build().unwrap().model(model_id.to_owned());
            let tokenizer_filename = repo.download("tokenizer.json").unwrap();//config.json, tokenizer.json,model.safetensors
            let tokenizer = Tokenizer::from_file(tokenizer_filename).unwrap();
            return tokenizer;
        }
    }
}
