# -*- coding: utf-8 -*-
""" utils/datasets/patchcamelyon """

import json
import os

import settings
from utils.datasets.mixins import BaseDBHandlerMixin


class DBhandler(BaseDBHandlerMixin):
    """
    Handler for PathCamelyon dataset

    The format of the arrays are [features, samples]

    Usage:
        train_feats, train_labels, test_feats, test_labels = DBhandler()()

    Source: https://github.com/giussepi/INCREMENTAL-LC-KSVD/blob/master/utils/datasets/patchcamelyon.py
    """

    def __init__(self, verbose=False):
        """ Loads the training and testing datasets """
        if verbose:
            print("Loading training dataset")
        with open(settings.TRAINING_DATA_DIRECTORY_DATASET_PATH, 'r') as file_:
            self.training_data = json.load(file_)
            self.to_numpy(self.training_data)

        if verbose:
            print("Loading testing dataset")
        with open(settings.TESTING_DATA_DIRECTORY_DATASET_PATH, 'r') as file_:
            self.testing_data = json.load(file_)
            self.to_numpy(self.testing_data)


class FeatsHandler(BaseDBHandlerMixin):
    """
    Handler for generated PatchCamelyon spatial pyramid features

    The format of the arrays are [features, samples]

    Usage:
        train_feats, train_labels, test_feats, test_labels = FeatsHandler()()

    Source: https://github.com/giussepi/INCREMENTAL-LC-KSVD/blob/master/utils/datasets/patchcamelyon.py
    """

    def __init__(self, verbose=False):
        """ Loads the training and testing datasets """
        filepath = os.path.join(
            settings.GENERATED_DATA_DIRECTORY,
            settings.GENERATED_FEATS_FILENAME_TEMPLATE.format('train')
        )

        if verbose:
            print("Loading training dataset")
        with open(filepath, 'r') as file_:
            self.training_data = json.load(file_)
            self.to_numpy(self.training_data)

        filepath = os.path.join(
            settings.GENERATED_DATA_DIRECTORY,
            settings.GENERATED_FEATS_FILENAME_TEMPLATE.format('test')
        )

        if verbose:
            print("Loading testing dataset")
        with open(filepath, 'r') as file_:
            self.testing_data = json.load(file_)
            self.to_numpy(self.testing_data)
