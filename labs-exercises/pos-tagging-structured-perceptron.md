# Lab - Part-of-Speech tagging with the Structured Perceptron Algorithm

In this lab we will create a Part-of-Speech (PoS) tagger for English using the Structured Perceptron algorithm. 
In particular, we will train a PoS tagger on the [Brown corpus](http://clu.uni.no/icame/manuals/BROWN/INDEX.HTM) 
annotated with the [Universal Part-of-Speech Tagset](https://arxiv.org/abs/1104.2086). 

[Last time](pos-tagging-structured-perceptron.md) we made each tagging decision independently. In this lab we will make
each decision at the sequence level, i.e., by choosing the PoS tag for each word so that they collectively *maximize* 
the score of the sequence of labels for the whole sentence. Why does that matter? Let's have a look  at the 
following example:

| **PoS**:   |  | | | |     |        |
|:-------|:-----:|:-----------:|------|:------:|:--------------:|:-----------:|
| **Words**: | The | old | man | the | boat | . |

If we predict with a trained model using the simple averaged perceptron implementation with unigram features from the 
[previous lab](pos-tagging-perceptron.md) we get the following predictions: 

  
| **PoS**:   | DET | ***ADJ**      | ***NOUN** | DET |     NOUN      | .       |
|:-------|:-----:|:-----------:|------|:------:|:--------------:|:-----------:|
| **Words**: | The | old | man | the | boat | . |

Why is this the case? **NOUN** is the highest scoring label for the word 'man'. Therefore, when making the prediction
for this word *independently*, the model will make an error (still not convinced this is wrong? Read the sentence carefully!).
(Why is *ADJ** also the wrong label for the word 'old'?)
 
The idea of the structured perceptron is that it keeps track of several alternative hypotheses for sequences of labels:
the one contained in the example above contains 'locally' high scoring labels (ADJ, NOUN), but has a much lower 'global'
score compared to the (correct) sequence below:  

| **PoS**:   | DET | NOUN      | VERB | DET |     NOUN      | .       |
|:-------|:-----:|:-----------:|------|:------:|:--------------:|:-----------:|
| **Words**: | The | old | man | the | boat | . |


## Requirements
You need to download the Brown corpus first through NLTK if you don't have it already. 
Just execute the following in a Python CLI:

```python
import nltk
nltk.downlad('brown')
``` 

## Exercises


#### 1. Structured Perceptron Algorithm

Implement the structured perceptron algorithm. Use the first 1000/100/100 sentences for training/dev/test with <8 words.
You can re-use the implemented simple dataset reader: `athnlp/reader/brown_pos_corpus.py`. 

The algorithm is starkingly similar to the original perceptron algorithm; the two major differences though are:
1. You need to find the optimal (a.k.a. *arg,max*) path through the input sequence; 
2. You need to update the weights for each label where the optimal predicted path and ground truth don't agree. 

Things to try out:
- (Sanity check) What is the accuracy score of the averaged pecreptron algorithm using unigrams for this dataset?
- First implement *arg,max* using brute-force, i.e., explore all the possible labeled paths.
- (If you have time) Brute-force is a really inefficient approach to finding the optimal path. 
A better solution is to implement the Viterbi algorithm, an instance of dynamic programming (see 
[section 8.4.5](https://web.stanford.edu/~jurafsky/slp3/8.pdf) from the Jurafsky & Martin book for the pseudo-code).   

You should evaluate your models by computing the **accuracy** (i.e., number-of-correctly-labelled-words / total-number-of-labelled-words) on the dev set.







