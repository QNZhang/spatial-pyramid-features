# -*- coding: utf-8 -*-
""" spfs/utils/utils """

import os
import json
from collections import OrderedDict

import numpy as np
from gutils.numpy_.numpy_ import LabelMatrixManager
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from tqdm import tqdm

import settings


def using_quick_tests():
    """
    Returns True if the app has been set to use a reduced dataset for
    quick testing
    """
    return settings.QUICK_TESTS > 0


def get_uint8_image(img):
    """
    Returns a ndarray of type uint8

    Args:
        img     (np.ndarray): image loaded using numpy

    Return:
        image (ndarray with dtype np.uint8)
    """
    if 0 <= img.min() and img.max() <= 1:
        img = (img*255).round()

    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    return img


def get_histogram_intersection(histogram_x, histogram_y):
    """
    Calculates and returns the histogram intersection

    I(H^l_X, H^l_X) = \sum_{i=1}^D \min(H^l_X (i), H^l_Y (i))

    Args:
        histogram_x (np.ndarray): histogram x (long histogram or histogram at level l)
        histogram_y (np.ndarray): histogram y (long histogram or histogram at level l)

    Returns:
        histogram intersection (int)
    """
    return np.minimum(histogram_x, histogram_y).sum()


def apply_pca(dataset, pca=None):
    """
    Applies PCA to the provided dataset

    Args:
        dataset            (np.ndarray): numpy array [samples, features]
        pca (sklearn.decomposition.PCA): fitted sklearn PCA instance

    Returns:
        np.ndarray [samples, settings.PCA_N_COMPONENTS], sklearn.decomposition.PCA fitted instance
    """
    if pca:
        return pca.transform(dataset), pca

    pca = PCA(settings.PCA_N_COMPONENTS, random_state=settings.RANDOM_STATE)

    return pca.fit_transform(dataset), pca


def create_15_scene_json_files(num_per_class=-1):
    """
    Creates the train and test JSON dataset files for 15-Scene dataset

    Args:
        num_per_class (int): number of samples per class. Set it to -1 or 0 to use all the samples
    """
    assert isinstance(num_per_class, int)

    train_json_dir = os.path.dirname(settings.TRAINING_DATA_DIRECTORY_DATASET_PATH)
    test_json_dir = os.path.dirname(settings.TESTING_DATA_DIRECTORY_DATASET_PATH)

    for dataset, json_dir in zip(('train', 'test'), (train_json_dir, test_json_dir)):
        categories_path = os.path.join('scene_data', 'train')
        dataset_path = os.path.join('scene_data', dataset)
        categories = OrderedDict((idx, name) for idx, name in enumerate(os.listdir(categories_path)))
        formatted_data = dict(codes=[], labels=[])

        for idx, category in categories.items():
            category_path = os.path.join(dataset_path, category)
            category_samples = os.listdir(category_path)

            if num_per_class > 0:
                category_samples = category_samples[:num_per_class]

            code_paths = [os.path.join(category_path, image) for image in category_samples]
            formatted_data['codes'].extend(code_paths)
            formatted_data['labels'].extend([idx] * len(code_paths))

        with open(os.path.join(json_dir, 'scene_{}.json'.format(dataset)), 'w') as file_:
            json.dump(formatted_data, file_)


class FeaturesEvaluator:
    """
    Holds methods to evaluate the spatial pyramid features created using
    Linear Support Vector Classification
    """

    @staticmethod
    def apply_linear_svc(
            c=0.0009075999999999997, train_feats=None, train_labels=None, test_feats=None,
            test_labels=None, verbose=True
    ):
        """
        Evaluates the features using LinearSVC and returns its accuracy

        Args:
            c (float): regularization parameter
            train_feats (np.ndarray): training spatial pyramid features
            train_labels (np.ndarray): training labels
            test_feats (np.ndarray): testing spatial pyramid features
            test_labels (np.ndarray): testing labels
            verbose (bool): Whether or not print messages

        Returns:
            accuracy (float)
        """
        assert isinstance(c, float)
        assert isinstance(verbose, bool)
        if train_feats is not None:
            assert isinstance(train_feats, np.ndarray)
        if train_labels is not None:
            assert isinstance(train_labels, np.ndarray)
        if test_feats is not None:
            assert isinstance(test_feats, np.ndarray)
        if test_labels is not None:
            assert isinstance(test_labels, np.ndarray)

        # avoiding circular reference
        from spfs.utils.datasets.handlers import FeatsHandler  # pylint: disable=import-outside-toplevel

        if train_feats is None and train_labels is None and test_feats is None and \
           test_labels is None:
            train_feats, train_labels, test_feats, test_labels = FeatsHandler(verbose=True)()
            if verbose:
                print('train_feats shape {}, train_labels shape {}'.format(
                    train_feats.shape, train_labels.shape))
                print('test_feats shape {}, test_labels shape {}'.format(
                    test_feats.shape, test_labels.shape))

        clf = LinearSVC(random_state=settings.RANDOM_STATE, C=c)
        clf.fit(train_feats.T, LabelMatrixManager.get_1d_array_from_2d_matrix(train_labels))
        predict = clf.predict(test_feats.T)
        accuracy = np.mean(predict == LabelMatrixManager.get_1d_array_from_2d_matrix(test_labels))*100

        if verbose:
            print("Accuracy: {}".format(accuracy))

        return accuracy

    @classmethod
    def find_best_regularization_parameter(cls, lower_bound=0.000307, upper_bound=0.001, step=0.0000462):
        """
        Find the best regularization parameter in the provided interval

        Args:
            lower_bound (float): interval lower bound
            upper_bound (float): interval upper bound
            step        (float): step used when going through the interval
        """
        # avoiding circular reference
        from spfs.utils.datasets.handlers import FeatsHandler  # pylint: disable=import-outside-toplevel

        train_feats, train_labels, test_feats, test_labels = FeatsHandler(verbose=True)()
        best_values = dict(c=-1, acc=0)
        worst_values = dict(c=-1, acc=0)

        for c in tqdm(np.arange(lower_bound, upper_bound, step), desc='Trying several C values: '):
            accuracy = cls.apply_linear_svc(c, train_feats, train_labels, test_feats, test_labels, False)

            if worst_values['c'] == -1 or accuracy < worst_values['acc']:
                worst_values['c'] = c
                worst_values['acc'] = accuracy

            if accuracy > best_values['acc']:
                best_values['c'] = c
                best_values['acc'] = accuracy

        print("Best values: C {} and accuracy {}".format(best_values['c'], best_values['acc']))
        print("Worst values: C {} and accuracy {}".format(worst_values['c'], worst_values['acc']))
