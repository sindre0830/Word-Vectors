# Word-Vectors

## Overview
This repository implements different architectures for training word embeddings. The architectures include Continuous Bag-of-Words (CBOW), skip-gram, and Global Vectors for Word Representation (GloVe). The flake8 dataset is used as training data, while the Google Analogy dataset and the WordSim353 dataset is used for validating the word embeddings.
- [Continuous Bag-of-Words (CBOW)](source/architechtures/cbow.py) architecture implementation
- [Skip-gram](source/architechtures/skipgram.py) architecture implementation
- [Global Vectors for Word Representation (GloVe)](source/architechtures/glove.py) architecture implementation

## Setup
0. Install required python version **3.11**
1. Install required packages `pip install -r source/requirements.txt` (We recommend using virtual environment, follow guide under **Virtual Environment Setup** below and skip this step)
2. Run program `python source/main.py`

### Virtual Environment Setup
#### Windows
0. Get the package `pip install virtualenv`
1. Create a new empty instance of python environment `py -3.11 -m venv ./.venv`
2. Activate the environment `source .venv/Scripts/activate`
3. Install the packages required by this project `pip install -r source/requirements.txt`

#### Linux
0. Get the package `pip install virtualenv`
1. Create a new empty instance of python environment `python -m venv ./.venv`
2. Activate the environment `source .venv/bin/activate`
3. Install the packages required by this project `pip install -r source/requirements.txt`
