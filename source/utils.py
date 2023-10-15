from constants import (
    PROJECT_DIRECTORY_PATH
)

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


def get_model_progressbar(iter, epoch: int, max_epochs: int) -> tqdm.tqdm:
    """
    Generates progressbar for iterable used in model training.
    """
    width = len(str(max_epochs))
    progressbar = tqdm.tqdm(
        iterable=iter,
        desc=f"Epoch {(epoch + 1):>{width}}/{max_epochs}"
    )
    set_model_progressbar_prefix(progressbar)
    return progressbar


def set_model_progressbar_prefix(
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


def plot_loss_and_accuracy(loss_history: list[float], accuracy_history: list[float], data_directory: str):
    title = "Training Metrics over Epochs"
    filepath = os.path.join(PROJECT_DIRECTORY_PATH, "data", data_directory, "plots", f"{title}.png")

    epochs = range(1, len(loss_history) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    # plot loss
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss", color="red")
    line1, = ax1.plot(epochs, loss_history, "r-", label="Training Loss")
    ax1.tick_params(axis='y', labelcolor="red")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    # plot accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy", color="blue")
    line2, = ax2.plot(epochs, accuracy_history, "b-", label="Training Accuracy")
    ax2.tick_params(axis='y', labelcolor="blue")
    # combine legends
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left")

    fig.tight_layout(pad=3.0)
    plt.title(title)
    # save plot
    save_plot(filepath)
    plt.close()
