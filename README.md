# Rust Lighter

This project was started as my RUST exercise to abstract the Rust minimalist ML framework Candle (https://github.com/huggingface/candle) and introduce a more convenient way of programming neural network machine learning models. 

The behaviour is inspired by Python KERAS (https://keras.io) and the initial step based on the Rust-Keras-like code (https://github.com/AhmedBoin/Rust-Keras-Like). 

So let's call the project **Candle Lighter** &#128367;, because it helps to turn on the candle light and is even easier to implement.

Examples can be found below the **lib/examples/** directory.  

To use it as library just call 'cargo add candlelighter'

**Note:** It is by far not production ready and is only used for own training purposes. No warranty and liability is given. I am a private person and not targeting any commercial benefits. 


# Supported Layer types

| Type         |      State    |  Example      | 
|--------------|---------------|---------------|
| Dense (aka feed forward network, short FFN)        |  &#9989;      | DNN           |
| Convolution  |  &#9989;      | CNN           |
| Pooling      |  &#9989;      | -             |
| Normalization|  &#9989;      | -             |
| Flatten      |  &#9989;      | -             | 
| Recurrent    |  &#9989;      | RNN 1st throw |  
| Regulation   |  &#9989;      | -             | 
| [Embedding](./docs/embedding.MD)     |  &#9989;      | S2S 1st throw |
| [Attention](./docs/attention.MD)    |  &#x1F3C3;    | TNN 1st throw  |
| Masking      |  &#x1F3C3;    | -             |
| Merging      |  &#x1F3C3;    | -             | 
| Fine tuning      |  &#x1F3C3;    | -             | 


Notes:
- **Masking** here is about handling sequences with varying lengths. 
- In comparison **Merging** refers to the concept of *ensembling learning* that combines multiple models to create a stronger and more robust one (aka model merging). 
- **Fine tuning** is about updating only selected neural network parameter e.g., while improving a given *foundation model*. Plan is to provide a DoRA and a LoRA implementation via Reinforcement Learning (RL, aka Transformer Reinforcement Learning, short TRL). 