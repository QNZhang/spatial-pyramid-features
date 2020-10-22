# -*- coding: utf-8 -*-
""" utils/normalization """

import numpy as np


class Normalization:
    """ Holds several normalization methods """

    @staticmethod
    def lp(vector, p):
        """
        Returns the lp normalized vector

        Args:
            vector (np.ndarray): vector to normalize
            p             (int): type of normalization

        Returns:
            np.ndarray
        """
        assert isinstance(vector, np.ndarray)
        assert isinstance(p, int)
        assert 0 <= p <= 2

        lp_norm = np.linalg.norm(vector, p)

        if lp_norm in (0, 1, np.nan):
            return vector

        return vector/lp_norm

    @classmethod
    def l0(cls, vector):
        """
        Returns the l0 normalized vector

        Args:
            vector (np.ndarray): vector to normalize

        Returns:
            np.ndarray
        """
        return cls.lp(vector, 0)

    @classmethod
    def l1(cls, vector):
        """
        Returns the l1 normalized vector

        Args:
            vector (np.ndarray): vector to normalize

        Returns:
            np.ndarray
        """
        return cls.lp(vector, 1)

    @classmethod
    def l2(cls, vector):
        """
        Returns the l2 normalized vector

        Args:
            vector (np.ndarray): vector to normalize

        Returns:
            np.ndarray
        """
        return cls.lp(vector, 2)

    @staticmethod
    def standard(vector):
        """
        Returns normalized vector following the formula: (vector - mean) / std

        Args:
            vector (np.ndarray): vector to normalize

        Returns:
            np.ndarray
        """
        dev = np.std(vector)
        return (vector-np.mean(vector))/dev
