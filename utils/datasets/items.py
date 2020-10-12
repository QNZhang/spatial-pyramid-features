# -*- coding: utf-8 -*-
""" utils/datasets/items """

import numpy as np
from PIL import Image

import settings
from utils.datasets.templates import DatasetItemsTemplate


class InMemoryDatasetItems(DatasetItemsTemplate):
    """ Handles a dataset by loading it into memory """

    def __init__(self, *args, **kwargs):
        """
        Initializes the object instance

        Args:
            1st (list of lists): dataset features e.g.: [[features1], [features2], ...]
        """
        assert isinstance(args[0], list)
        assert isinstance(args[0][0], list)

        self.dataset = np.array(args[0])

    @property
    def num_samples(self):
        """ Returns the number of samples """
        return self.dataset.shape[1]

    def get_sample(self, index):
        """
        Returns a 1D numpy array containing the values/features of the sample located at
        postition index

        Args:
            index (int): sample position in the dataset
        """
        assert isinstance(index, int)

        return self.dataset[:, index].reshape(settings.IMAGE_WIDTH, settings.IMAGE_HEIGHT)


class LazyDatasetItems(DatasetItemsTemplate):
    """ Handles a dataset by loading images into memory only when required """

    def __init__(self, *args, **kwargs):
        """
        Initializes the object instance

        Args:
            1st (list of strings): image paths e.g.: [[path1], [path2], ...]
        """
        assert isinstance(args[0], list)
        assert isinstance(args[0][0], str)

        self.dataset = args[0]
        self.datase_num_samples = len(self.dataset)

    @property
    def num_samples(self):
        """ Returns the number of samples """
        return self.datase_num_samples

    def get_sample(self, index):
        """
        Returns a 2D numpy array containing the values/features of the sample located at
        postition index.

        If the image does not have the gray-scale PIL format then it is converted, before
        returning it as a numpy array

        Args:
            index (int): sample position in the dataset
        """
        assert isinstance(index, int)

        img = Image.open(self.dataset[index])

        return np.asarray(img.convert('L')) if img.getbands() != ('L', ) else np.asarray(img)
