import utils

import sys


def main():
    (error, cmd) = utils.parse_arguments(args=sys.argv[1:])
    print(error, cmd)


if __name__ == "__main__":
    main()
