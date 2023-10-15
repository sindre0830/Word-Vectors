from constants import (
    PROJECT_DIRECTORY_PATH
)
import utils
import datahandler.loaders
import models

import os
import numpy as np
import torch


def run() -> None:
    config = utils.load_config(filepath=os.path.join(PROJECT_DIRECTORY_PATH, "source", "config_skipgram.yml"))
    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    # get corpus
    corpus = datahandler.loaders.Corpus()
    corpus.build(
        pipeline=[
            corpus.download,
            corpus.flatten
        ],
        data_directory="skipgram"
    )
    # get vocabulary
    vocabulary = datahandler.loaders.Vocabulary(add_padding=True, add_unknown=False)
    vocabulary.build(corpus.words, config.vocabulary_size)
