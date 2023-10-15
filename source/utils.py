import os
import yaml
import argparse
import numpy as np
import requests


def load_config(filepath: str) -> argparse.Namespace:
    # read config file
    with open(filepath, 'r') as file:
        config_dict: dict = yaml.load(file, Loader=yaml.FullLoader)
    # store config parameters to Namespace object
    config = argparse.Namespace()
    for key, value in config_dict.items():
        setattr(config, key, value)

    return config


def parse_arguments(args: list[str]) -> str:
    args_size = len(args)
    if (args_size <= 0):
        return None
    return args[0]


def print_commands() -> None:
    msg = "\nList of commands:\n"
    msg += "\t'--help' or '-h': \tShows this information\n"
    msg += "\t'--cbow' or '-cbow': \tStarts the CBOW program, takes parameters from 'config_cbow.yml' file\n"
    print(msg)


def print_operation(message):
    """Print operation that allows status message on the same line."""
    print('{:<60s}'.format(message), end="", flush=True)


def print_operation_status(message: str = "DONE"):
    """Print message in console."""
    print(message)


def print_divider():
    """Print divider in console."""
    print("\n")


def save_numpy(filepath: str, object: np.ndarray):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.save(filepath, object)


def load_numpy(filepath: str) -> np.ndarray:
    return np.load(filepath)


def download_file(url: str, save_path: str):
    if os.path.exists(save_path):
        return
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    response = requests.get(url, allow_redirects=True)
    # ensure the request was successful
    response.raise_for_status()
    
    with open(save_path, 'wb') as file:
        file.write(response.content)
