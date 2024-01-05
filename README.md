# Rust Lighter

This project was started as my RUST exercise to abstract the Rust minimalist ML framework Candle (https://github.com/huggingface/candle) and introduce a more convenient way of programming neural network machine learning models. 

The behaviour is inspired by Python KERAS (https://keras.io) and based on the Rust-Keras-like (https://github.com/AhmedBoin/Rust-Keras-Like) code. 

The Rust Candle ecosystem facilitates terms that refer to candles &#128367;. So let's call the project **Candle Lighter** &#9617;, because it helps to turn on the candle light and is even easier to implement.

An example can be found below *src/main.rs*.  

**Note:** It is by far not production ready and is only used for own training purposes. No warranty and liability is given. I am a private person and not targeting any commercial benefits. 


# Supported Layer types

| Type         |      State    |  
|--------------|:-------------:|
| Dense        |  &#9989;      | 
| Convolution  |  &#9989;      |   
| Pooling      |  &#9989;      |   
| Normalization|  &#9989;      |   
| Regulation   |  &#10062;     | 
| Embedding    |  &#10062;     | 
| Masking      |  &#10062;     | 
| Reshaping    |  &#10062;     | 
| Merging      |  &#10062;     | 
| Activation   |  &#10062;     | 
| Recurrent    |  &#10062;     |   
| Attention    |  &#10062;     |   