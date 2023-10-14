import utils

import sys


def main():
    cmd = utils.parse_arguments(args=sys.argv[1:])
    
    match cmd:
        case "--help" | "-h":
            utils.print_commands()
            return
        case _:
            print("Error: Unknown command.")
            utils.print_commands()


if __name__ == "__main__":
    main()
