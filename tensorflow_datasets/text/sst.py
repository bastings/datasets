# coding=utf-8
# Copyright 2019 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The Stanford Sentiment Treebank (SST)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_CITATION = """
@inproceedings{socher-etal-2013-recursive,
    title = "Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank",
    author = "Socher, Richard  and
      Perelygin, Alex  and
      Wu, Jean  and
      Chuang, Jason  and
      Manning, Christopher D.  and
      Ng, Andrew  and
      Potts, Christopher",
    booktitle = "Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing",
    month = oct,
    year = "2013",
    address = "Seattle, Washington, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D13-1170",
    pages = "1631--1642",
}
"""

_DESCRIPTION = """
The Stanford Sentiment Treebank (SST) consists of 11,855 parse trees of 
single-sentence movie reviews, annotated with sentiment scores at each node.
The task is to predict the sentiment for a sentence or tree.

The data set provides five fine-grained labels: very negative, negative,
neutral, positive, very positive.
Every node in the tree has been annotated with such a sentiment label.

The provided trees correspond to the constituent tree experiment in 
Tai et al., "Improved Semantic Representations From 
Tree-Structured Long Short-Term Memory Networks", ACL 2015.

We provide tree structure as shift-reduce transition sequences so that a 
binary tree can be constructed, as well as the sentiment label for each node.
The sequence of sentiment labels for just the words is also provided.
"""

_DOWNLOAD_URL = "https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip"

_LABELS = ['very negative', 'negative', 'neutral', 'positive', 'very positive']
_TRANSITIONS = ["shift", "reduce"]


class SST(tfds.core.GeneratorBasedBuilder):
  """The Stanford Sentiment Treebank (SST)."""

  VERSION = tfds.core.Version('1.0.0')

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'label': tfds.features.ClassLabel(names=_LABELS),
            'text': tfds.features.Text(),
            'tags': tfds.features.Sequence(
                tfds.features.ClassLabel(names=_LABELS)),
            'transitions': tfds.features.Sequence(
                tfds.features.ClassLabel(names=_TRANSITIONS)),
            'node_labels': tfds.features.Sequence(
                tfds.features.ClassLabel(names=_LABELS)),
        }),
        supervised_keys=('text', 'label'),
        urls=["https://nlp.stanford.edu/sentiment/index.html"],
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""

    dl_dir = dl_manager.download_and_extract(_DOWNLOAD_URL)
    data_dir = os.path.join(dl_dir, 'trees')

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            num_shards=1,
            gen_kwargs={"path": os.path.join(data_dir, "train.txt")}),
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            num_shards=1,
            gen_kwargs={"path": os.path.join(data_dir, "dev.txt")}),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            num_shards=1,
            gen_kwargs={"path": os.path.join(data_dir, "test.txt")}),
    ]

  @staticmethod
  def _get_label(score):
    """Returns the sentence label as text given a numerical score 0-4."""
    assert 0 <= score < 5, "fine-grained score must be in {0, 1, 2, 3, 4}"
    return _LABELS[score]

  @staticmethod
  def _get_text(s):
    """Extracts the text (leafs) from a tree string."""
    return " ".join(re.findall(r"\([0-9] ([^\(\)]+)\)", s))

  @staticmethod
  def _get_tags(s):
    """Extracts the sequence of tags (pre-terminals) from a tree string."""
    scores = map(int, re.findall(r"\(([0-9]) [^\(\)]", s))
    return [SST._get_label(score) for score in scores]

  @staticmethod
  def _get_node_labels(s):
    """Extracts the sequence of node labels from a tree string."""
    scores = map(int, re.findall(r"\(([0-9]) ", s))
    return [SST._get_label(score) for score in scores]

  @staticmethod
  def _get_transitions(s):
    """Returns a sequence of shift/reduce operations from a tree string."""
    s = re.sub(r"\([0-5] ([^)]+)\)", "shift", s)
    s = re.sub(r"\)", " )", s)
    s = re.sub(r"\([0-4] ", "", s)
    s = re.sub(r"\([0-4] ", "", s)
    s = re.sub(r"\)", "reduce", s)
    return s.split()

  def _generate_examples(self, path):
    """Yields examples."""
    with tf.io.gfile.GFile(path) as f:

      for example_id, s in enumerate(f):
        s = re.sub("\\\\", "", s)

        label_id = int(s[1])
        label = SST._get_label(label_id)

        text = SST._get_text(s)
        tags = SST._get_tags(s)

        transitions = SST._get_transitions(s)
        node_labels = SST._get_node_labels(s)

        yield example_id, {"label": label,
                           "text": text,
                           "tags": tags,
                           "transitions": transitions,
                           "node_labels": node_labels}
