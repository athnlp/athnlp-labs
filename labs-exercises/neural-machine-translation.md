# Lab - Neural Machine Translation


## Introduction

In this lab we will familiarise ourselves with the popular sequence-to-sequence (seq2seq) architecture for Neural Machine 
Translation and will implement the attention mechanism.  

## Requirements
We will train our models on the Multi30k dataset (well just a small part of it, as we will be 
running things on your laptop; you are more than welcome to try out the full dataset on a GPU-enabled machine too!).

1. Clone the dataset from [here](https://github.com/multi30k/dataset). 
2. Extract the first 1000 examples of the already tokenized version of the validation set: 
``data/task1/tok/val.lc.norm.tok.*``. 
3. Create a 75\%/25\% train/val split. 
4. We will focus only on the ``en-fr`` pairs. 

## Exercises

We provide an implementation of a basic sequence-to-sequence (seq2seq) architecture with beam search 
adapted from the original AllenNLP toolkit that you will have to extend: ``athnlp/models/nmt_seq2seq.py``. 
There are placeholders in the code that are left empty for you to fill in. We are also giving you 
a dataset reader for Multi30k: ``athnlp/readers/multi30k_reader.py``. 

**Note**: We recommend that
you train and predict with the built-in commands using ``allennlp train/predict``. If you
need to debug your code you can programmatically execute the training process from: ``athnlp/nmt.py``
We will be reporting performance using [BLEU](https://www.aclweb.org/anthology/P02-1040).   

#### 1. Playing around

Have a good look at the provided code and make sure you understand how it works.

Things to try out:

- Overfit a (very) small portion of the training set. What hyperparameters do you need to use?
- Train a model on the bigger dataset for a few epochs and compute BLEU score for the baseline model. 
**Note**: You are most likely not going to get a state-of-the-art performance. Why?
- Switch the RNN cell from an LSTM to a GRU.
- Use pre-trained embeddings like [GloVe](https://nlp.stanford.edu/pubs/glove.pdf) vectors. Does it help? 
Is that always applicable in MT?
- Consider switching the metric (currently it's the validation loss) for early stopping criterion.
- Try using beam search instead of greedy decoding. Does it help?
 
  

#### 2. Attention Mechanism

Implement at least one attention mechanism ([dot product](https://arxiv.org/abs/1508.04025), 
[bilinear](https://arxiv.org/abs/1508.04025), [MLP](https://arxiv.org/abs/1409.0473)) 
in the methods ``_prepare_output_projections()`` and ``_compute_attention()``. 

**Important**: to keep things uniform assume that the attended encoder outputs 
(aka *context vector*) gets concatenated with the previous predicted word embedding *before* being 
fed as in input to the decoder RNN. 

Things to try out:

- Convince yourself that attention helps boost the performance of your model by computing
BLEU on the dev set (if not most probably you have a bug!)
- Predict the output for some examples using the default ``se2seq`` predictor from AllenNLP. 
You can find a small set of examples here: ``data/multi30k/val.lc.norm.tok.head-5.fr.jsonl``. 
How does the output compare to without using attention?
- Visualise the attention scores using e.g., ``matplotlib.heatmap``. We have created a custom predictor
in ``athnlp/predictors/nmt_seq2seq.py`` for you that already prints out heatmaps; you will just need to extract the attention
scores from your model in ``forward_loop()``. You can execute it via ``athnlp/nmt.py``. 
 Then visualize attention scores between the input and predicted output for the examples 
 found in ``data/multi30k/val.lc.norm.tok.head-5.fr.jsonl``. What do you observe?
  
- (Bonus) Instead of concatenating the context vector with the previous predicted word embedding *before* being 
feeding as in input to the decoder RNN, try concatenating it to the output hidden state of the decoder 
(i.e., *after* the RNN). Does that effect any change? (**Note**: you might need to train on the original corpus). 
    
#### 3. (Bonus) Sampling during decoding
Implement a sampling algorithm for your decoder. As an alternative heuristic to beam search during decoding, 
the idea is to *sample* from the vocobulary distribution instead of taking the ``arg_max`` at each time step.

Things to try out:

- How does sampling affect the performance (BLEU score) of your model?
- Inspect the output of your model by drawing several samples for a few examples; what do you observe? 
   
