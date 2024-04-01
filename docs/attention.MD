We will support different attention approaches. [Candle](https://github.com/huggingface/candle) provides us a brought varierty on existing implementations. 

| Type                  | From where                                                                                        |  
|-----------------------|---------------------------------------------------------------------------------------------------|
| SelfAttention         | Integrated- [here](https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/wuerstchen/attention_processor.rs) for one input sequence. Depending on the implementation it is also called a *dot product attention* or *global attention*.| 
| CrossAttention (aka Co-Attention)       | Not Integrated so far - [here](https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/stable_diffusion/attention.rs) for multiple input sequences | 
| CausalSelfAttention   | Not Integrated so far - [here](https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/llama2_c.rs) for parts of one or multiple input sequences e.g., only all token before the present. Depending on the implementation also called *local attention*. | 
| MultiHeadAttention   | Not Integrated so far - [here](https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/segment_anything/transformer.rs) for multiple concerns/ questions | 
| MultiQueryAttention   | Not Integrated so far - [here](https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/chatglm.rs) for multiple concerns/ questions but knowing the other concerns/ questions | 
| GroupQueryAttention   | Not Integrated so far - [here](https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/quantized_mpt.rs) for building logical groups between the questions | 


*Terms*:
- Heads: amount on parallel questions on a given stream.
- Contexts: amount of parallel streams.
- Temporal: Time.
- Spatial: Dimensionality. 

*Note: All attention should be available for multiple dimensions. This includes spatial transformer which acts in >= 2D space (=spatial) as required for CNN applications.*

More complex models mappes as own layer:
- https://arxiv.org/html/2312.06635v3
    * [here](https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/rwkv_v6.rs) 