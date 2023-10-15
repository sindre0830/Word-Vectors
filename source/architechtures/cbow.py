from constants import (
    PROJECT_DIRECTORY_PATH
)
import utils
import datahandler.loaders

import os
import numpy as np
import torch


def run() -> None:
    config = utils.load_config(filepath=os.path.join(PROJECT_DIRECTORY_PATH, "source", "config_cbow.yml"))
    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    # get corpus
    corpus = datahandler.loaders.Corpus()
    corpus.build(
        pipeline=[
            corpus.download,
            corpus.flatten
        ]
    )
    # get vocabulary
    vocabulary = datahandler.loaders.Vocabulary(add_padding=True, add_unknown=False)
    vocabulary.build(corpus.words, config.vocabulary_size)
    # get training data
    training_dataloader = datahandler.loaders.DataLoaderCBOW(config.batch_size)
    training_dataloader.build(corpus.sentences, vocabulary, config.window_size, config.device)
    # get validation data
    validation_dataloader = datahandler.loaders.ValidationLoader()
    validation_dataloader.build(vocabulary, data_directory="cbow")
