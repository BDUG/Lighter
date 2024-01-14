# Rust Lighter

This project was started as my RUST exercise to abstract the Rust minimalist ML framework Candle (https://github.com/huggingface/candle) and introduce a more convenient way of programming neural network machine learning models. 

The behaviour is inspired by Python KERAS (https://keras.io) and based on the Rust-Keras-like (https://github.com/AhmedBoin/Rust-Keras-Like) code. 

The Rust Candle ecosystem facilitates terms that refer to candles &#128367;. So let's call the project **Candle Lighter** &#9617;, because it helps to turn on the candle light and is even easier to implement.

Examples can be found below the **lib/examples/** directory.  

To use it as library just call 'cargo add candlelighter'

**Note:** It is by far not production ready and is only used for own training purposes. No warranty and liability is given. I am a private person and not targeting any commercial benefits. 


# Supported Layer types

| Type         |      State    |  Example      | 
|--------------|---------------|---------------|
| Dense        |  &#9989;      | DNN           |
| Convolution  |  &#9989;      | CNN           |
| Pooling      |  &#9989;      | -             |
| Normalization|  &#9989;      | -             |
| Flatten      |  &#9989;      | -             | 
| Recurrent    |  &#9989;      | RNN 1st throw |  
| Regulation   |  &#9989;      | -             | 
| Embedding    |  &#x1F3C3;    | S2S pending   |
| Attention    |  &#x1F3C3;    | -             |
| Masking      |  &#x1F3C3;    | -             |
| Merging      |  &#x1F3C3;    | -             | 