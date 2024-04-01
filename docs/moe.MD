For the same sequence, a Mixture of Experts (MoE) model would use a gating mechanism to **decide** which 'expert' is best suited to handle the prediction at each point in the sequence. This implies it takes one out of many experts. In comparison e.g. Multi-Head Attention is about splitting focus to **simultaneously** capture different types of relationships in the data. It's like having multiple lenses to look at the data from different angles at once. 

From the parallelism perspective MoE does not have a bottleneck, because the sequence can be splittet into simultanous processed snippets of the input. A Multi-Head Attention whoudl require a syncronization point (merge).

Example:

Input sequence: 1 2 3 4 - 5 6 + 8 9
Expected outcome: 4-5 = - 1 and 6+8= 14

MoE: 2 results -1 and 14, because of different tasks in a sequence 
Attention: Might be 13, because an sense within the given sequence. 

This is a classical MoE case. We will have an expert for the **minus** and one for **plus** operation. 


Fields of application:

**MoE**
- Distributing/ Sharding of large sequences
- Heterogenous data which can be distinguid into subsets 
- Different tasks to perform

**Multi Head Attention**
- Capture and attent different, complex dependencies and relationships within a stream 
- Cross and multi modal data sequences

We will support different attention approaches. [Candle](https://github.com/huggingface/candle) provides us a brought varierty on existing implementations. 

| Type                  | From where                                                                                        |  
|-----------------------|---------------------------------------------------------------------------------------------------|
| SparseMoe         | Not integrated, because of own implementation - Can be found in Candle [here](https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/qwen2_moe.rs) and [here](https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/mixtral.rs). Only run a few tasks (aka top k experts) and teh other will be *nulled*. |
| Multi Task Moe         | **N/A** in Candal. Run multiple tasks in parallel and collects each results individually. |
| Gated Moe         | **N/A** in Candle. Not only performance an expert selection, but also adapts the trainign process. |
| Hierachical Moe         | **N/A** in Candle. Stacks experts to evolutionary determine a result. |
| Conditional Moe         | **N/A** in Candle. Also know as "Switch MoE". Purpose is the selection of at least one expert for a specifi task. |

*Note:*: Sparse (aka Sparsamkeit).