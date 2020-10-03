# -*- coding: utf-8 -*-
""" utils/datasets/patchcamelyon """

import json
import os

from sklearn.model_selection import train_test_split
from gutils.numpy_ import format_label_matrix

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
        # create a subdataset using a percentage and load it
        FeatsHandler().create_subsets(percentage=20, verbose=True)
        train_feats, train_labels, test_feats, test_labels = FeatsHandler(percentage=20, verbose=True)()

    Inspired on:
        https://github.com/giussepi/INCREMENTAL-LC-KSVD/blob/master/utils/datasets/patchcamelyon.py
    """

    def __init__(self, percentage=0, verbose=False):
        """ Loads the training and testing datasets """
        assert isinstance(percentage, int)

        if 0 < percentage < 100:
            train_path = settings.GENERATED_FEATS_FILENAME_TEMPLATE.format(
                '{}%_train'.format(percentage))
            test_path = settings.GENERATED_FEATS_FILENAME_TEMPLATE.format(
                '{}%_test'.format(percentage))
        else:
            train_path = settings.GENERATED_FEATS_FILENAME_TEMPLATE.format('train')
            test_path = settings.GENERATED_FEATS_FILENAME_TEMPLATE.format('test')

        filepath = os.path.join(settings.GENERATED_DATA_DIRECTORY, train_path)

        if verbose:
            print("Loading training dataset")
        with open(filepath, 'r') as file_:
            self.training_data = json.load(file_)
            self.to_numpy(self.training_data)

        filepath = os.path.join(settings.GENERATED_DATA_DIRECTORY, test_path)

        if verbose:
            print("Loading testing dataset")
        with open(filepath, 'r') as file_:
            self.testing_data = json.load(file_)
            self.to_numpy(self.testing_data)

    def create_subsets(self, percentage=20, verbose=False):
        """
        Creates subsets of the training and testing datasets considering provided percentage

        Args:
            percentage (int): percentage of the dataset to use
            verbose   (bool): whether to pring messages or not
        """
        assert 0 < percentage < 100

        train_feats, train_labels, test_feats, test_labels = self()

        filepath = os.path.join(
            settings.GENERATED_DATA_DIRECTORY,
            settings.GENERATED_FEATS_FILENAME_TEMPLATE.format('{}%_train'.format(percentage))
        )

        if verbose:
            print("Saving training dataset subset at {}".format(filepath))

        with open(filepath, 'w') as file_:
            xtrain, _, ytrain, _ = train_test_split(
                train_feats.T, train_labels[1], train_size=percentage/100,
                random_state=settings.RANDOM_STATE, shuffle=True,
                stratify=train_labels[1]
            )
            formatted_data = dict(
                codes=xtrain.T.tolist(),
                labels=format_label_matrix(ytrain).tolist()
            )
            json.dump(formatted_data, file_)

        filepath = os.path.join(
            settings.GENERATED_DATA_DIRECTORY,
            settings.GENERATED_FEATS_FILENAME_TEMPLATE.format('{}%_test'.format(percentage))
        )

        if verbose:
            print("Saving testing dataset subset at {}".format(filepath))

        with open(filepath, 'w') as file_:
            xtrain, _, ytrain, _ = train_test_split(
                test_feats.T, test_labels[1], train_size=percentage/100,
                random_state=settings.RANDOM_STATE, shuffle=True,
                stratify=test_labels[1]
            )
            formatted_data = dict(
                codes=xtrain.T.tolist(),
                labels=format_label_matrix(ytrain).tolist()
            )
            json.dump(formatted_data, file_)
