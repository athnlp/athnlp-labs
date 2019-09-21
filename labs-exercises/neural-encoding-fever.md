# Lab - Neural Encoding for Text Classification

## Introduction 

This lab will introduce continuous representations for NLP. 
We will work on the task of Natural Language Inference (also known as Textual Entailment) in the context of the Fact Extraction and Verification dataset introduced by [Thorne et al. (2018)](https://arxiv.org/abs/1803.05355). 
We will focus on the subtask of deciding whether a claim is supported or refuted given a set of evidence sentences. 
The dataset also contains claims for which no appropriate evidence was found in Wikipedia; we will ignore these in this lab.

To simplify the task, we have prepared a _Lite_ version of the dataset that has the Wikipedia evidence bundled together with each dataset instance. The full task requires searching for this evidence.

## Requirements

- Use a subset of the FEVER data (provided in `data/fever/`) to predict whether textual _claims_ are SUPPORTED or REFUTED from _evidence_ from Wikipedia. More details about how it was prepared can be found [here](https://github.com/j6mes/feverlite/releases).
- Use the [AllenNLP](https://allennlp.org/) framework for implementation of your neural models. **(If you installed the required Python packages for the summer school, AllenNLP should be installed for you).**
- It is highly recommended to use an IDE, such as PyCharm, for working in this project


## AllenNLP Primer
There are four key parts you'll interact with when developing with the AllenNLP framework:

* Dataset Reader
* Model
* Configuration File
* Command Line Interface / Python Module

### Dataset Reader and Sample Data
Each labeled dataset instance consists of a `claim` sentence accompanied by one or more `evidence` sentences. 

```
{    
    'label': 'SUPPORTS',
    'claim': 'Ryan Gosling has been to a country in Africa.',
    'evidence': [
        'He is a supporter of PETA , Invisible Children and the Enough Project and has traveled to Chad , Uganda and eastern Congo to raise awareness about conflicts in the regions .', 
        "Chad -LRB- -LSB- tʃæd -RSB- تشاد ; Tchad -LSB- tʃa -LRB- d -RRB- -RSB- -RRB- , officially the Republic of Chad -LRB- ; `` Republic of the Chad '' -RRB- , is a landlocked country in Central Africa ."
    ]
}
```

We provide code to read through the dataset files in `athnlp/readers/fever_reader.py`. 
The dataset reader we provide does all the necessary preprocessing before we pass the data to the model. For example, in our implementation, we tokenize the sentences.
 
This returns an `Instance` that consists of a `claim` and `evidence` for the model. 
Notice that the instance contains a `TextField` for the tokenized sentences and a `LabelField` for the label. 
The framework will construct a vocabulary using the words in the TextField for you. 
If you want to add hand-crafted features, this might be a good place to add them (you could add an array of features in an `ArrayField`).

Also notice that above the `FEVERLiteDatasetReader` there is a decorator `@DatasetReader.register("feverlite")`. This will come in handy when using configuration files for our model as it associates the type `feverlite` with this class -- this will come in handy later!


### Model
Just like we registered our dataset reader, we can also register a `Model`. In the file `athnlp/models/fever_text_classification.py`, we have built a skeleton model that you can adapt for the exercises. We have registered it with the name `fever` by using the decorator `@Model.register("fever")` above the class name. If you plan on adding more models, you should think of a unique name.

The model has a `forward(...)` method: this is the main method for prediction just like we'd expect to find with other models written in `PyTorch`. 
Notice how in our model, the argument names match up with the values returned by the dataset reader: AllenNLP will match these up for you during training and model evaluation. 
While the variable names are the same, the data types are different. AllenNLP will convert a `TextField` into a LongTensor - each element in this tensor corresponds to the index of the token in the vocabulary.
AllenNLP will automatically generate batches for you: this means that all variables here are batch-first tensors.

The model returns quite a bit of information to the trainer that is calling it. 
It is quite common to see the following code in a lot of AllenNLP models.
The loss is computed by the model (if a `label` is passed in) and this is what is used for error backpropagation.
If we need to compute any metrics, such as accuracy or F1 score, this would be the place to do it.
```
        label_probs = F.softmax(label_logits, dim=-1)
        output_dict = {"label_logits": label_logits,
                       "label_probs": label_probs}

        if label is not None:
            loss = self._loss(label_logits, label.long().view(-1))
            self._accuracy(label_logits, label)
            output_dict["loss"] = loss
            
        return output_dict
```

The meat of the model will perform a sequence of operations to the input data, returning the label logits and loss.
It is possible to mix torch and AllenNLP operations

Operations that might be helpful for the exercises are:

* Embedding Lookup (TextFieldEmbedder)
* Feed-forward Neural Networks (FeedForward)
* Summing tensors (torch.sum())
* Concatenating tensors (torch.cat())

  
### Configuration
Parameters for the model are stored in a JSON file. For this example, you can adapt `athnlp/experiments/fever.json`. 
In this configuration file, there are separate configurations for the `datasetreader`, `model` and `trainer`. 
Notice how the `type` of the datasetreader and model match the values we specified earlier. 

The values in this configuration are passed to the constructor of our model and dataset reader and also match their parameters:

```json
 "model": {
    "type": "fever",
    "text_field_embedder": {
      ...
    },
    "final_feedforward": {
      ...
    },
    "initializer": [
      ...
     ]
  }
```
And in the python code for the model, the `__init__` method of the model takes these parameters. Note that `vocab` is auto-filled by another part of AllenNLP.
You can find examples of configs from real-world models on [GitHub](https://github.com/allenai/allennlp/tree/master/training_config).
```python
@Model.register("fever")
class FEVERTextClassificationModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 final_feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator()):
 ```
 
### Running the model
AllenNLP will install itself as a bash script that you can call when you want to train/evaluate your model using the config specified in your json file. Using the `--include-package` option will load the custom models, dataset readers and other Python modules in that package.

```bash
allennlp train --force --include-package athnlp --serialization-dir mymodel myconfig.json
``` 

`train` tells AllenNLP to train the model, there are other options for Fine Tuning, Evaluation, Predicting etc.
`--serialization-dir` defines the location the model will be saved
`--force` will overwite any existing model saved in the serialization-dir. You could use `--recover` if you wish to continue training a model from a checkpoint.
`--include-package` will import the Python described package here.

This is an alias that just runs Python with the following command: `python -m allennlp.run [args]`. 
If you are using an IDE, you can debug AllenNLP models by running the python module `allennlp.run`. **note: this is running a module. not a script - select the dropdown to select "Module name" NOT "Script path"**

![](/data/run_fever.png)

If you are using `pdb`, you will have to write a simple 2-line wrapper script: 
```
from allennlp.commands import main 
main(prog="allennlp”)
```

### Debugging

#### ConfigurationError

If you encounter this error:
```
allennlp.common.checks.ConfigurationError: "feverlite not in acceptable choices for dataset_reader.type: ['ccgbank', 'conll2003', 'conll2000', 'ontonotes_ner', 'coref', 'winobias', 'event2mind', 'interleaving', 'language_modeling', 'multiprocess', 'ptb_trees', 'drop', 'squad', 'quac', 'triviaqa', 'qangaroo', 'srl', 'semantic_dependencies', 'seq2seq', 'sequence_tagging', 'snli', 'universal_dependencies', 'universal_dependencies_multilang', 'sst_tokens', 'quora_paraphrase', 'atis', 'nlvr', 'wikitables', 'template_text2sql', 'grammar_based_text2sql', 'quarel', 'simple_language_modeling', 'babi', 'copynet_seq2seq', 'text_classification_json']"
```
Check that `--include-package athnlp` is included in the arguments when calling AllenNLP


#### ModuleNotFoundError

If you encounter this error: 
```
ModuleNotFoundError: No module named 'athnlp'
```
Check that the athnlp folder is in the `PYTHONPATH`. Is your current working directory the `athnlp-labs` folder?

#### FileNotFoundError

```
FileNotFoundError: file resources/glove.6B.50d.txt.gz not found
```

Run setup_dependencies.sh to download the file (it will then call `wget https://allennlp.s3.amazonaws.com/datasets/glove/glove.6B.50d.txt.gz -P resources/;`)


#### NotImplementedError
This is where you should add your solution to the exercise! Go and edit `athnlp/models/fever_text_classification.py` and delete this line.
```
NotImplementedError: Compute label logits (for supported and refuted) for the given Claim and Evidence input
```


#### It is running all my scripts!
If you have put your code in the `athnlp`, the `--include-package` function will try and find code to import. You should change the code from previous labs and wrap it in the following if statement `if __name__ == "__main__":` so that it only runs when it is your main python script.  

### Self-Help
There are a large number of (more complex) models already available for AllenNLP which could help inspire you when you write your model: Check out the [models package](https://github.com/allenai/allennlp/tree/master/allennlp/models) on GitHub for inspiration if you are stuck. 

If you are getting errors about size mismatch (`RuntimeError: size mismatch, m1: [32 x 100], m2: [256 x 100]`) check the dimensions of your MLP are compatible. This error is caused when PyTorch tries to multiply incompatible matrices. Check that the input dimension for the MLP in the config file is the same size as the input representation you generate. Use the debugger to inspect this or print the shape of your variables `print(my_variable.shape)`. 


### Using GPU
If your laptop has a CUDA-enabled GPU and you have the appropriate drivers installed, you can speed up training by setting `"cuda_device":0` in your configuration file. 

## Exercises
For the exercises, we have provided a dataset reader (`athnlp/readers/fever_reader.py`), configuration file (`athnlp/experiments/fever.json`), and sample model (`athnlp/models/fever_text_classification.py`). You can complete these exercises by completing the code in the sample model.

### 1. Average Word Embedding Model
1. Implement a model that 
	- represents the claim and the evidence by averaging their word embeddings;
	- concatenates the two representations;
	- uses a multilayer perceptron to decide the label.

2. Experiment with the number and the size of hidden layers to find the best settings using the train/dev set and assess your accuracy on the test set. (note: this model may not get high accuracy)

3. Explore: How does fine-tuning the word embeddings affect performance? You can make the word embeddings layer trainable by changing the config file for the `text_field_embedder` in the `fever.json` config file. 

### 2. Discrete Feature Baseline
Start by make a new config file and a new model file based on `fever.json` and `fever_text_classification.py`. Don't forget  


1. Compare against a discrete feature baseline. Instead of embedding claim and evidence we are making an n-hot Bag of Words vector. (Hint: edit the type of the `text_field_embedder` be `bag_of_word_counts` - you will have make changes to your model too!).

2. How does limiting the vocabulary size affect the model accuracy?  (hint: adding this to the main section in the config file will limit the vocab size to 10000 tokens. `"vocabulary":{"max_vocab_size":10000}`)


### 3. Convolution
Averaging word embeddings is an example of a CBOW model. An alternative way to combine the representations is to use CNNs (see slide 110/111 in Ryan McDonald's talk: [SLIDES](https://github.com/athnlp/athnlp-labs/blob/master/slides/McDonald_classification.pdf)).

1. Use a `CnnEncoder()` ([documentation](https://allenai.github.io/allennlp-docs/api/allennlp.modules.seq2vec_encoders.html#allennlp.modules.seq2vec_encoders.cnn_encoder.CnnEncoder)) to generate convoluted sentence representations. (debugging hint: this method expects the input to be padded. you may get errors if filter size is longer than the sentnece. you will need to set `"token_min_padding_length": 5` or higher in the `tokens` object in `token_indexers` for large filter sizes). Filter sizes of between 2-5 should be sufficient. More filters will cause training to be slower (perhaps just train for 1 or 2 epochs). 

### 4. Hypothesis-Only NLI and Biases
1. Implement a _[hypothesis only](https://www.aclweb.org/anthology/S18-2023)_ version of the model that ignores the evidence and only uses the claim for predicting the label. What accuracy does this model get? Why do you think this? Think back to slide 7 on Ryan's talk. 

2. Take a look at the training/dev data. Can you design claims that would "fool" your models? You can see this report ([Thorne and Vlachos, 2019](https://arxiv.org/abs/1903.05543)) for inspiration. 
What do you conclude about the ability of your model to understand language?
