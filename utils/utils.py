# -*- coding: utf-8 -*-
""" utils/utils """

import numpy as np
from sklearn.decomposition import PCA

import settings


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


def apply_pca(dataset):
    """
    Applies PCA to the provided dataset

    Args:
        dataset (np.ndarray): numpy array [samples, features]

    Returns:
        np.ndarray [samples, settings.PCA_N_COMPONENTS]
    """
    return PCA(settings.PCA_N_COMPONENTS, random_state=settings.RANDOM_STATE).fit_transform(dataset)
