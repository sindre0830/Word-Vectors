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


def parse_arguments(args: list[str]) -> tuple[str, str]:
    error = None
    cmd = None
    args_size = len(args)
    if (args_size <= 0):
        error = "No arguments given"
        return (error, cmd)
    cmd = args[0]
    return (error, cmd)
