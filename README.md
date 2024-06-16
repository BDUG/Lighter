# Rust Lighter

This project was started as my RUST exercise to abstract the Rust minimalist ML framework Candle (https://github.com/huggingface/candle) and introduce a more convenient way of programming neural network machine learning models. 

The behaviour is inspired by Python KERAS (https://keras.io) and the initial step based on the Rust-Keras-like code (https://github.com/AhmedBoin/Rust-Keras-Like). 

So let's call the project **Candle Lighter** &#128367;, because it helps to turn on the candle light and is even easier to implement.

Examples can be found below the **lib/examples/** directory.  

To use it as library just call 'cargo add candlelighter'

**MAINTAINERS AND CONTRIBUTORS ARE HIGHLY WELCOME**


**Note:** It is by far not production ready and is only used for own training purposes. No warranty and liability is given. I am a private person and not targeting any commercial benefits. 


# Supported Layer types

| Meta Layer | Type         |      State    |  Example      | 
|-----| --------------|---------------|---------------|
| Sequential model | - |   &#9989;     |     |
| - | Feature scaling      |  &#x1F3C3;     | [DNN](./lib/examples/simple_dnn.rs) and [TNN](./lib/examples/simple_tnn.rs)             |
| - | Dense        |  &#9989;      | [DNN](./lib/examples/simple_dnn.rs)           |
| - | Convolution  |  &#9989;      | [CNN](./lib/examples/simple_cnn.rs)           |
| - | Pooling      |  &#9989;      | -             |
| - | Normalization|  &#9989;      | -             |
| - | Flatten      |  &#9989;      | -             |  
| - | Recurrent    |  &#9989;      | [RNN](./lib/examples/simple_rnn.rs) 1st throw |  
| - | Regulation   |  &#9989;      | -             | 
| - | Recurrent    |  &#9989;      | [RNN](./lib/examples/simple_rnn.rs) 1st throw |  
| - | [Autoencoder](./docs/autoencoder.MD)     | &#x1F3C3;    | -             | 
| - | [Feature embedding](./docs/embedding.MD)     |  &#9989;      | [S2S](./lib/examples/simple_s2s.rs) 1st throw |
| - | [Attention](./docs/attention.MD)    |  &#x1F3C3;    | [TNN](./lib/examples/simple_tnn.rs) 1st throw  |
| - | [Mixture of Experts](./docs/moe.MD)   |  &#x1F3C3;    | [ENN](./lib/examples/simple_enn.rs) 1st throw             |
| - |  [Feature masking and -quantization](./docs/masking.MD)   |  &#x1F3C3;    | -             |
| - |  [KAN-Dense](https://www.holeoftherabbit.com/2024/06/16/may-kan-will-be-the-next-ai-disruption-step/)   |  &#x1F3C3;    | -             |
| [Model fine tuning  (PEFT)](https://www.holeoftherabbit.com/2024/06/14/fine-tuning-as-playfield/)  |  -  |  &#x1F3C3;    | *In development:* [DNN2](./lib/examples/simple_dnn.rs) & [DNN3](./lib/examples/simple_dnn.rs)            | 
| Parallel model (in sense of split) |  -   |  &#x1F3C3;    | [PNN](./lib/examples/simple_pnn.rs) 1st throw           | 
| Parallel model |  [Merging](./docs/modelmerging.MD)      |  &#x1F3C3;    | [PNN](./lib/examples/simple_pnn.rs) 1st throw             | 
| Transformer models |  [see](./docs/transformers.MD)     |  &#x1F3C3;    |          | 
| * BERT |  Text similarity    |  &#9989;  |   [LLM](./lib/examples/simple_llm.rs)       | 
| * LLAMA |  Completion (Chat)    |  &#9989;  |   [LLM2](./lib/examples/simple_llm.rs)       | 
| Reinforcement models |  [see](./docs/reinforcement.MD)     |  &#x1F3C3;    |          | 


# License
Tripple-licensed to be compatible with the Rust project and the source roots.

Licensed under the [MPL 2.0](./LICENSE), [MIT license](http://opensource.org/licenses/MIT) or the [Apache license, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0) at your option. 