from constants import (
    PROJECT_DIRECTORY_PATH
)
import utils

import os
import numpy as np
import torch


def run() -> None:
    config = utils.load_config(filepath=os.path.join(PROJECT_DIRECTORY_PATH, "source", "config_cbow.yml"))
    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
