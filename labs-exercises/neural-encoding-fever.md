### Lab - Neural Encoding for Text Classification

#### Introduction 

This lab will introduce continuous representations for NLP. We will work on the task of Natural Language Inference (also known as Textual Entailment) in the context of the Fact Extraction and Verification dataset introduced by [Thorne et al. (2018)](https://arxiv.org/abs/1803.05355). We will focus on the subtask of deciding whether a claim is supported or refuted given a set of evidence sentences. The dataset also contains claims for which no appropriate evidence was found in Wikipedia; we will ignore these in this lab.

#### Requirements

- Use a subset of the FEVER data as described in the paper:
    - [Train/Dev](https://s3-eu-west-1.amazonaws.com/fever.public/paper_dev.jsonl) sets
(suggested split: 75%:25%, but choice is yours)
    - [Test](https://s3-eu-west-1.amazonaws.com/fever.public/paper_test.jsonl) set

- GloVe word embeddings ([the small ones trained on Wikipedia](http://nlp.stanford.edu/data/glove.6B.zip)) should come in handy.


#### Exercises

1. Implement a model that 
	- represents the claim and the evidence by averaging their word embeddings;
	- concatenates the two representations;
	- uses a multilayer perceptron to decide the label.
Experiment with the number and the size of hidden layers to find the best settings using the train/dev set and assess your accuracy on the test set.
2. Compare against a discrete feature baseline, i.e., using one-hot vectors instead of word embeddings to represent the words?
3. Take a look at the training/dev data. Can you design claims that would "fool" your models? You can see this report ([Thorne and Vlachos, 2019](https://arxiv.org/abs/1903.05543)) for inspiration. 
What do you conclude about the ability of your model to understand language?

<!--stackedit_data:
eyJoaXN0b3J5IjpbLTExNzYxMDMzMDddfQ==
-->