#!/usr/bin/env bash
conda create -n athnlp -y;
conda activate athnlp;

pip install -r requirements.txt;

python -m nltk.downloader brown;
