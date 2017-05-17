"""Grid search parameter sweep

version 2
    using parameters and os.system
"""

import os


def gridsearch(search):
    """Perform a grid-search on a different script"""

    for i in search["num_layers"]:
        for j in search["state_size"]:
            for k in search["run"]:
                args = "--num_layers=%d --state_size=%d --run=%d" % (i, j, k)
                os.system("python conv_ca.py " + args)

    print("grid search complete")


def main():
    """Set grid search options and start search"""
    search = {"num_layers": {1},
              "state_size": {1},
              "run": {99, 98}}
    gridsearch(search)

if __name__ == '__main__':
    main()
