import yaml
import argparse


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
