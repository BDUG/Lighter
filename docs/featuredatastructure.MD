The hardest part in machine learning often revolves around ensuring data is in the correct shape for various machine learning models and operations. This is because machine learning algorithms are highly sensitive to the shape of the input data. Common challenges are:

- **Understanding Data Representation**: Knowing how data should be structured for different types of machine learning tasks can be complex. Each type of data and model architecture might require a unique arrangement of dimensions.
- **Batch Size Dimension**: Many machine learning frameworks expect the first dimension of your data array to represent the batch size. Remembering to include this dimension, even if you're processing a single sample (batch size of 1), can be a source of error.
- **Channels or Features Dimension**: For image data, it's crucial to know the expected order of dimensions, particularly the position of the channels dimension (e.g., RGB channels in an image). Some frameworks expect channels first (channels, height, width), while others expect channels last (height, width, channels).
- **Time Series and Sequences**: For sequential data, such as time series or text, managing the sequence length dimension can be challenging. Different models and parts of the same model might require this dimension to be in different places or to be of a specific length, sometimes necessitating padding or truncation of sequences.
- **Broadcasting Rules**: Understanding how arrays with different shapes are treated in operations (like addition, multiplication) according to broadcasting rules can be tricky. Inadvertently broadcasting arrays can lead to unexpected results or performance issues.
- **Reshaping and Squeezing**: Knowing when and how to reshape (change the dimensionality) or squeeze/unsqueeze (adding or removing singleton dimensions) without losing the meaningful structure of your data requires a good understanding of your data and the model's expectations.

And many more aspects...

With Lighter this should become a bit easier. A helper *class* will support to describe the intended input features dimensionality:
- *1st dimension (**rational**):* Samples, at least one
- *2nd dimension (**temporal**):* Time steps, at least one
- *3rd dimension (**spatial**):* Features in different dimensions, at least a 1 dimensional sequence 

Per definition *batches* are not taken under consideration. This kind of shuffling operation we move to the *model* fitting consideration. The model fittign will consider multiple given samples for the batching. 