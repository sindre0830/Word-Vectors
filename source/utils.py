import os
import yaml
import argparse
import numpy as np
import requests
import matplotlib.pyplot as plt
import tqdm


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
    print()


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


def normalize(x: np.ndarray, axis = None, keepdims = False) -> np.ndarray:
    return x / np.linalg.norm(x, axis=axis, keepdims=keepdims)


def cosine_similarity(x_1: np.ndarray, x_2: np.ndarray):
    return np.dot(x_1, x_2)


def save_plot(filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, format="png")


def set_progressbar_prefix(
        progressbar: tqdm.tqdm,
        train_loss: float = 0.0,
        best_loss: float = 0.0,
        train_acc: float = 0.0,
        best_acc: float = 0.0
    ):
    """
    Set prefix in progressbar and update output.
    """
    train_loss_str = f"loss: {train_loss:.5f}"
    best_loss_str = f"best loss: {best_loss:.5f}"
    train_acc_str = f"acc: {train_acc:.5f}"
    best_acc_str = f"best acc: {best_acc:.5f}"
    progressbar.set_postfix_str(f"{train_loss_str}, {best_loss_str}, {train_acc_str}, {best_acc_str}")
