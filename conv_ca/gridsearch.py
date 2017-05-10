"""File that grid searches model for certain parameters"""

import conv_ca  # pylint: disable=import-error


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
            fn("num_layers", k)
            for j in search["state_size"]:  # state
                fn("state_size", j)
                for i in search["run"]:  # runs
                    fn("run", i)
                    conv_ca.main()
    finally:
        print("grid search complete")


def main():
    """Set grid search options and start search"""
    search = {"num_layers": {1, 2},
              "state_size": {3, 4},
              "run": {i for i in range(1, 5)}}
    gridsearch(search)

if __name__ == '__main__':
    main()
