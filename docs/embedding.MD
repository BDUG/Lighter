Both embedding ( e.g., Word2Vec) and encoding (e.g., Bag of words) is about representing data in a different space. Embedding isually talks about continous vector spaces (aka sequences), usually capturing semantic relationships, where encoding also includes compressing and dimensional reduction. 

The following embeddings shall be integrated from [Candle](https://github.com/huggingface/candle):


| Type                  | From where                                                                                        |  
|-----------------------|---------------------------------------------------------------------------------------------------|
| Embedding         | Standard layer | 
| Timestep Embedding         | [here](https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/stable_diffusion/embeddings.rs) for relative (aka local) positional encoding |  
| Positional Embedding         | [here](https://github.com/huggingface/candle/blob/0c5eecbc0faa7e642210800c735ad8137d5a9e08/candle-transformers/src/models/segment_anything/prompt_encoder.rs#L25) for absolute positional encoding | 
| Falcon Rotary Positional Embedding         | [here](https://github.com/huggingface/candle/blob/0c5eecbc0faa7e642210800c735ad8137d5a9e08/candle-transformers/src/models/falcon.rs#L163) for absolute and relative positional encoding | 
| Sinusoidal Positional Embedding         | [here](https://github.com/huggingface/candle/blob/0c5eecbc0faa7e642210800c735ad8137d5a9e08/candle-transformers/src/models/marian.rs#L108) for absolute and relative positional encoding | 



