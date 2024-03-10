We will support different attention approaches. [Candle](https://github.com/huggingface/candle) provides us a brought varierty on existing implementations. 

| Type                  | From where                                                                                        |  
|-----------------------|---------------------------------------------------------------------------------------------------|
| SelfAttention         | [here](https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/rwkv_v6.rs) for one input sequence. Depending on the implementation also called *dot product attention*. Depending on the respective implementation also called *global attention*.| 
| CrossAttention (aka Co-Attention)       | [here](https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/stable_diffusion/attention.rs) for multiple input sequences | 
| CausalSelfAttention   | [here](https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/llama2_c.rs) for parts of one or multiple input sequences e.g., only all token before the present. Depending on the implementation also called *local attention*. | 
| MultiHeadAttention   | [here](https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/segment_anything/transformer.rs) for multiple concerns/ questions | 
| MultiQueryAttention   | [here](https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/chatglm.rs) for multiple concerns/ questions but knowing the other concerns/ questions | 
| GroupQueryAttention   | [here](https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/quantized_mpt.rs) for building logical groups between the questions | 


*Note: All attention should be available for multiple dimensions. This includes spatial transformer which acts in >= 2D space (=spatial) as required for CNN applications.*