from constants import (
    PROJECT_DIRECTORY_PATH
)
import utils

import os


def run() -> None:
    config = utils.load_config(filepath=os.path.join(PROJECT_DIRECTORY_PATH, "source", "config_cbow.yml"))
    print(config.seed)
