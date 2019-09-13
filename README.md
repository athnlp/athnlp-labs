# ΑθNLP 2019

Exercises for the lab sessions of ΑθNLP 2019. 
The labs will cover the following:

1. [Part-of-Speech Tagging with the Perceptron algorithm](labs-exercises/pos-tagging-perceptron.md)
2. [Part-of-Speech Tagging with the Structured Perceptron algorithm](labs-exercises/pos-tagging-structured-perceptron.md)
3. [Neural Encoding for Text Classification](labs-exercises/neural-encoding-fever.md)
4. [Neural Language Modeling](labs-exercises/neural-language-model.md)
5. [Neural Machine Translation](labs-exercises/neural-machine-translation.md)
6. [Question Answering](labs-exercises/question-answering.md)

## Setup

You will need to have Python 3 installed on your machine; we recommend using [Anaconda](https://www.anaconda.com/), 
which is available for the most common OS distributions. 

For the first two labs we will be using vanilla Python (along with the standard scientific libarires, i.e., NumPy, SciPy, 
etc), while for the rest we will additionally be using [PyTorch](https://pytorch.org/) and 
[AllenNLP](https://allennlp.org/).

Use the Anaconda command-line tools to create a new virtual environment with Python 3.6:
```
    conda create --name athnlp python=3.6
```
After the installation is complete, you should have a new virtual environment called `athnlp` in your Anaconda installation 
that you can *activate* using the following command: `conda activate athnlp`. Remember to execute this command before
running the scripts in this repository.

We provide a script that will help you setup your environment. Run the command: `sh setup_dependencies.sh` and 
it will automatically install the project dependencies for you. The script will download several data dependencies that might
require some time to be installed. 


**Note**: Installing AllenNLP on Mac OS can be tricky; check [here](https://stackoverflow.com/questions/52509602/cant-compile-c-program-on-a-mac-after-upgrade-to-mojave)
for a possible solution.



