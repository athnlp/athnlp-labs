#!/usr/bin/env bash

conda activate athnlp;

pip install -r requirements.txt;

python -m nltk.downloader brown;

mkdir resources;

# We download in advance all the models/data that are required by AllenNLP and BERT
wget -c https://allennlp.s3.amazonaws.com/datasets/glove/glove.6B.50d.txt.gz -P resources/;

mkdir resources/bert-base-uncased;

wget -c https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt -O resources/bert-base-uncased/vocab.txt;

wget -c https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin -O resources/bert-base-uncased/pytorch_model.bin;

wget -c "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json" -O resources/bert-base-uncased/config.json;
