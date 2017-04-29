"""File that grid searches model for certain parameters"""

# from conv_ca import conv_ca_model
from conv_ca import FLAGS

function_map = {
    "num_layers": FLAGS.num_layers,
    "state_size": FLAGS.state_size
}

gridsearch = {"num_layers": {1, 2},
              "state_size": {1, 2}}


def run_model(arg):
    return "runtime"


i = len(gridsearch)

while i > 0:
    i -= 1

    for field, values in gridsearch.items():
        print (field, values)

        print("starting run for %s" % (field, values))
        try:
            metrics = run_model(FLAGS)
        finally:
            # report_metrics(metrics)
            pass

# def gridsearch(fields, values):
