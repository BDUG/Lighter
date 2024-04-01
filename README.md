# Rust Lighter

This project was started as my RUST exercise to abstract the Rust minimalist ML framework Candle (https://github.com/huggingface/candle) and introduce a more convenient way of programming neural network machine learning models. 

The behaviour is inspired by Python KERAS (https://keras.io) and the initial step based on the Rust-Keras-like code (https://github.com/AhmedBoin/Rust-Keras-Like). 

So let's call the project **Candle Lighter** &#128367;, because it helps to turn on the candle light and is even easier to implement.

Examples can be found below the **lib/examples/** directory.  

To use it as library just call 'cargo add candlelighter'

**CONTRIBUTORS ARE HIGHLY WELCOME**


**Note:** It is by far not production ready and is only used for own training purposes. No warranty and liability is given. I am a private person and not targeting any commercial benefits. 


# Supported Layer types

| Meta Layer | Type         |      State    |  Example      | 
|-----| --------------|---------------|---------------|
| Sequential model | - |   &#9989;     |     |
| - | Feature scaling      |  &#x1F3C3;     | DNN and TNN             |
| - | Dense        |  &#9989;      | DNN           |
| - | Convolution  |  &#9989;      | CNN           |
| - | Pooling      |  &#9989;      | -             |
| - | Normalization|  &#9989;      | -             |
| - | Flatten      |  &#9989;      | -             | 
| - | Recurrent    |  &#9989;      | RNN 1st throw |  
| - | Regulation   |  &#9989;      | -             | 
| - | [Feature embedding](./docs/embedding.MD)     |  &#9989;      | S2S 1st throw |
| - | [Attention](./docs/attention.MD)    |  &#x1F3C3;    | TNN 1st throw  |
| - | [MoE switch](./docs/moe.MD)   |  &#x1F3C3;    | ENN 1st throw             |
| - |  [Feature masking and -quantization](./docs/masking.MD)   |  &#x1F3C3;    | -             |
| Parallel model (in sense of split) |  -   |  &#x1F3C3;    | -             |
| Parallel model |  [Merging](./docs/modelmerging.MD)      |  &#x1F3C3;    | -             | 
| - |  [Model fine tuning](./docs/finetuning.MD)      |  &#x1F3C3;    | -             | 


# License
Tripple-licensed to be compatible with the Rust project and the source roots.

Licensed under the [MPL 2.0](./LICENSE), [MIT license](http://opensource.org/licenses/MIT) or the [Apache license, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0) at your option. 