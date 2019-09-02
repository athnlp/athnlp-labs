#!/usr/bin/env bash

conda activate athnlp;

pip install -r requirements.txt;

python -m nltk.downloader brown;