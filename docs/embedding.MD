Both embedding and encoding is about representing data in a different space. Embedding isually talks about continous vector spaces (aka sequences), capturing semantic relationships, where encoding also includes compressing and dimensional reduction. 

For *candlelighter* it is target to introduce processing pipelines allowing to formalizing the preprocesing and the finalizing embedding step. The embedding step maps each dimension element from the input vector to the target, resulting vector, while each resultign vector often has a higher dimensionality. 

The preprocessing usually covers multiple steps such as e.g., vectorization, dimensional reduction and positional enconding. Technically it is like the sequentional model, but without the backward step support. 