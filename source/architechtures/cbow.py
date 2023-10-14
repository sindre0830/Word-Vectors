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
    corpus_loader = datahandler.loaders.CorpusLoader()
    corpus_loader.build(
        pipeline=[
            corpus_loader.download,
            corpus_loader.flatten
        ]
    )
