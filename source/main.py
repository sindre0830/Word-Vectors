from constants import (
    PROJECT_DIRECTORY_PATH
)
import utils

import os


def main():
    config_filepath = os.path.join(PROJECT_DIRECTORY_PATH, "source", "config.yml")
    config = utils.load_config(config_filepath)
    print(config.model)


if __name__ == "__main__":
    main()
