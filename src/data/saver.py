"""Object to load and save hdf5 files"""

import os
import h5py


class Saver(object):
    def __init__(self, path):
        self.path = path

    def load_hdf5(self):
        with h5py.File(self.path, "r") as h5file:
            examples = h5file["examples"]
            labels = h5file["labels"]

        return examples[:], labels[:]

    def save_hdf5(self, examples, labels):

        mdir = os.path.dirname(self.path)
        if not os.path.exists(mdir):
            os.makedirs(mdir)

        with h5py.File(self.path, "w") as h5file:
            dset_examples = h5file.create_dataset("examples", data=examples, dtype="i", compression="gzip")
            dset_examples.attrs["Description"] = "Examples"
            dset_labels = h5file.create_dataset("labels", data=labels, dtype="i", compression="gzip")
            dset_labels.attrs["Description"] = "Labels"
