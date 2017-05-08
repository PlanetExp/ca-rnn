"""File that grid searches model for certain parameters"""

import conv_ca

s = {"num_layers": {1, 2},
     "state_size": {3, 4}}


def fn(f, v):
    if f == "num_layers":
        conv_ca.FLAGS.num_layers = v
    elif f == "state_size":
        conv_ca.FLAGS.state_size = v
    elif f == "max_steps":
        conv_ca.FLAGS.max_steps = v


try:
    fn("max_steps", 50000)
    fn("num_layers", 6)
    fn("state_size", 3)
    conv_ca.main()
    fn("num_layers", 5)
    conv_ca.main()
    fn("num_layers", 6)
    conv_ca.main()
    fn("num_layers", 7)
    conv_ca.main()
finally:
    print("grid search complete")
