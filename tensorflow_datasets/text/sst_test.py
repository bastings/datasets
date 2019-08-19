"""TODO(sst): Add a description here."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_datasets import testing
from tensorflow_datasets.text import sst


class SSTTest(testing.DatasetBuilderTestCase):

  DATASET_CLASS = sst.SST
  SPLITS = {
      "train": 3,
      "validation": 1,
      "test": 1,
  }


if __name__ == "__main__":
  testing.test_main()
