# Lab - Neural Language Modeling


## Introduction

In this lab we will create a Language Model using Recurrent Neural Networks with PyTorch. 

## Requirements
We will train our model on the following toy dataset:

```
<s> The thief stole . </s>
<s> The thief stole the suitcase . </s>
<s> The crook stole the suitcase . </s>
<s> The cop took a bribe . <s>
<s> The thief was arrested by the detective . </s>
```

## Exercises


#### 1. Language Modeller

Implement a LSTM-based RNN language model that takes each word of a sentence as input and
predicts the next one (the original RNNLM demo paper can be found 
[here](http://www.fit.vutbr.cz/~imikolov/rnnlm/rnnlm-demo.pdf)).   
In particular, the input to the RNN is the previous word and the previous hidden state and the output is the next 
predicted word. 

**Note**: Consider each sentence as a separate example, where each sentence is represented as a list of tokens.

Things to try out:
- Run a sanity check: make sure your model can learn how to predict correctly your training data. After training your
model, take the sentence
    ```
    <s> The thief stole the suitcase . </s>
    ```
    and check that for every word and context (i.e., last hidden state of the RNN) you get the right answer. Does it work?
    For example, given the context ``<s> The`` the model should be predicting ``thief``. 
    Why is this happening instead of predicting ``crook``? 
    
    **Note**: You might need to play with the hyper-parameters, such as learning rate, epoch number etc.
     
#### 2. Sentence Completion

Given a sentence with a gap
```
<s> The ______ was arrested by the detective . </s>
```
implement a decoder that returns the most likely word to fill it in. 
In more detail, you can develop a k-best ranker that scores the top-k derivations that a) all start with the prefix 
``The``, b) each contains the top-k candidate words from the vocabulary, and c) follow with the rest words of the given
sentence.

Things to try out:
- Which is more likely to fill in the gap: ``cop`` or ``crook``? 
Get the model to predict this correctly by changing the hyper-parameters. 
- Ensure that the model is predicting correctly for the right reason, 
i.e., that the embeddings for ``thief`` and ``crook`` are closer to each other than the embeddings 
for ``thief`` and ``cop``. Why is that? 

    **Hint**: Use cosine similarity to compute the distance of two embedding vectors.

