"""File that grid searches model for certain parameters"""

import conv_ca  # pylint: disable=import-error
import os

def fn(f, v):
    """links a string to functions of imported script"""
    if f == "num_layers":
        conv_ca.FLAGS.num_layers = v
    elif f == "state_size":
        conv_ca.FLAGS.state_size = v
    elif f == "max_steps":
        conv_ca.FLAGS.max_steps = v
    elif f == "run":
        conv_ca.FLAGS.run = v


def gridsearch(search):
    """Perform a grid-search on a different script"""
    try:
        # fn("max_steps", 50000)
        for k in search["num_layers"]:  # layers
            # fn("num_layers", k)
            # num_layers = "--num_layers=%d" % k
            for j in search["state_size"]:  # state
                # fn("state_size", j)
                # state_size = "--state_size=%d" % j
                for i in search["run"]:  # runs
                    # fn("run", i)
                    args = "--num_layers=%d --state_size=%d --run=%d" % (k, j, i)
                    # conv_ca.main()
                    os.system("python conv_ca.py " + args)
    finally:
        print("grid search complete")


def main():
    """Set grid search options and start search"""
    search = {"num_layers": {1},
              "state_size": {1},
              "run": {99, 98}}
    gridsearch(search)

if __name__ == '__main__':
    main()
