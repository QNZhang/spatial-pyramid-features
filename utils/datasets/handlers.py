# -*- coding: utf-8 -*-
""" utils/datasets/handlers """

import json
import os

import numpy as np
from gutils.numpy_.numpy_ import LabelMatrixManager
from sklearn.model_selection import train_test_split

import settings
from utils.datasets.items import InMemoryDatasetItems, LazyDatasetItems
from utils.datasets.mixins import BaseDBHandlerMixin, DataTransformsMixin
from utils.utils import using_quick_tests


class BaseDBHandler(BaseDBHandlerMixin):
    """
    Base database handler (NOT TO BE USED DIRECTLY)



    The shape of the returned lables arrays are

    Usage:
        train_feats, train_labels, test_feats, test_labels = DBhandler()()

    Inspired on: https://github.com/giussepi/INCREMENTAL-LC-KSVD/blob/master/utils/datasets/patchcamelyon.py

    Returns:
        feats (list of lists)
        labels (np.ndarray) with shape [num labels, num samples]
    """

    def __init__(self, verbose=False):
        """ Loads the training and testing datasets """
        for db_split, db_path, attr_name in (
                ('training', settings.TRAINING_DATA_DIRECTORY_DATASET_PATH, 'training_data'),
                ('testing', settings.TESTING_DATA_DIRECTORY_DATASET_PATH, 'testing_data')
        ):
            if verbose:
                print("Loading {} dataset".format(db_split))

            with open(db_path, 'r') as file_:
                data = json.load(file_)

            setattr(self, attr_name, dict())

            # if it's not a 2D label matrix coded as JSON, then turn it into a 2D matrix
            if not isinstance(data['labels'][0], list):
                getattr(self, attr_name)['labels'] = LabelMatrixManager.get_2d_matrix_from_1d_array(
                    np.array(data['labels']))
            else:
                getattr(self, attr_name)['labels'] = np.array(data['labels'])

            # QuickTests
            if using_quick_tests():
                getattr(self, attr_name)['labels'] = \
                    getattr(self, attr_name)['labels'][:, :settings.QUICK_TESTS]

            getattr(self, attr_name)['codes'] = data['codes']


class InMemoryDBHandler(BaseDBHandler):
    """
    Memory-intensive dataset handler

    Handles the codes by loading them into memory using a single 2D matrix

    Usage:
        from core.feature_extractors import SpatialPyramidFeatures
        from utils.datasets.handlers import InMemoryDBHandler

        train_feats, train_labels, test_feats, test_labels = InMemoryDBHandler()()

        spf = SpatialPyramidFeatures(LazyDBHandler)
        spf.create_codebook()
        spf.create_spatial_pyramid_features()


    Returns:
        training_data_codes (InMemoryDatasetItems): see class definiton
        training_data_labels          (np.ndarray): array with shape [feats, samples]
        testing_data_codes  (InMemoryDatasetItems): see class definiton
        testing_data_labels           (np.ndarray): array with shape [feats, samples]
    """

    def __init__(self, verbose=False):
        super().__init__(verbose)
        self.training_data['codes'] = InMemoryDatasetItems(self.training_data['codes'])
        self.testing_data['codes'] = InMemoryDatasetItems(self.testing_data['codes'])


class LazyDBHandler(BaseDBHandler):
    """
    Memory-efficent dataset handler

    Handles the codes, in a memory efficent way, by loading one at a time into memory only when
    it is required

    Usage:
        from core.feature_extractors import SpatialPyramidFeatures
        from utils.datasets.handlers import LazyDBHandler

        train_feats, train_labels, test_feats, test_labels = LazyDBHandler()()

        spf = SpatialPyramidFeatures(LazyDBHandler)
        spf.create_codebook()
        spf.create_spatial_pyramid_features()

    Returns:
        training_data_codes (LazyDatasetItems): see class definiton
        training_data_labels      (np.ndarray): array with shape [feats, samples]
        testing_data_codes  (LazyDatasetItems): see class definiton
        testing_data_labels       (np.ndarray): array with shape [feats, samples]
    """

    def __init__(self, verbose=False):
        super().__init__(verbose)
        self.training_data['codes'] = LazyDatasetItems(self.training_data['codes'])
        self.testing_data['codes'] = LazyDatasetItems(self.testing_data['codes'])


class FeatsHandler(DataTransformsMixin, BaseDBHandlerMixin):
    """
    Handler for generated spatial pyramid features

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
        Creates subsets of the spatial pyramid features training dataset considering
        the provided percentage as the percentage covered by the subset training dataset.
        Thus, the features training dataset is splitted into 'percentage'% for training
        and '100-percentage' % for testing.

        Args:
            percentage (int): percentage of the featues dataset to be used for the training subset
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
                labels=LabelMatrixManager.get_2d_matrix_from_1d_array(ytrain).tolist()
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
                labels=LabelMatrixManager.get_2d_matrix_from_1d_array(ytrain).tolist()
            )
            json.dump(formatted_data, file_)
