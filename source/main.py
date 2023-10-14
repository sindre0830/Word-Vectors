import utils
import architechtures.cbow

import sys


def main():
    cmd = utils.parse_arguments(args=sys.argv[1:])

    match cmd:
        case "--help" | "-h":
            utils.print_commands()
            return
        case "--cbow" | "-cbow":
            print("Starting CBOW program...\n")
            architechtures.cbow.run()
            return
        case _:
            print("Error: Unknown command.")
            utils.print_commands()
            return


if __name__ == "__main__":
    main()
