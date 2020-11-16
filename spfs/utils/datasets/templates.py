# -*- coding: utf-8 -*-
""" spfs/utils/datasets/templates """


class DatasetItemsTemplate:
    """ Basic structure for dataset items handlers """

    dataset = None

    def __init__(self, *args, **kwargs):
        """ Initializes the object instance """
        raise NotImplementedError

    @property
    def num_samples(self):
        """  Returns the number of samples """
        raise NotImplementedError

    def get_sample(self, index):
        """
        Returns a 1D numpy array containing the values/features of the sample located at
        postition index
        """
        raise NotImplementedError
