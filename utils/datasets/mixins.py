# -*- coding: utf-8 -*-
""" utils/datasets/mixins """

import numpy as np


class BaseDBHandlerMixin:
    """ Provides base methods to work with datasets """

    def __call__(self):
        """ functor call """
        return self.__get_training_testing_sets()

    @staticmethod
    def to_numpy(data):
        """  """
        for key in data:
            data[key] = np.array(data[key])

    def __get_training_testing_sets(self):
        """ Returns training and testing data """
        return self.training_data['codes'], self.training_data['labels'], \
            self.testing_data['codes'], self.testing_data['labels']
