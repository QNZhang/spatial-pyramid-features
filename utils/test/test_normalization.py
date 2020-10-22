# -*- coding: utf-8 -*-
""" utils/test/test_normalization """

import unittest

import numpy as np

from utils.normalization import Normalization


class TestNormalization(unittest.TestCase):

    def setUp(self):
        self.vector = np.arange(4)

    def test_lp(self):
        for p in range(3):
            Normalization.lp(self.vector, p)

        with self.assertRaises(AssertionError):
            Normalization.lp(self.vector, 3)

    def test_l0(self):
        self.assertTrue(
            np.array_equal(
                Normalization.l0(self.vector),
                self.vector/np.linalg.norm(self.vector, 0)
            )
        )

    def test_l1(self):
        self.assertTrue(
            np.array_equal(
                Normalization.l1(self.vector),
                self.vector/np.linalg.norm(self.vector, 1)
            )
        )

    def test_l2(self):
        self.assertTrue(
            np.array_equal(
                Normalization.l2(self.vector),
                self.vector/np.linalg.norm(self.vector, 2)
            )
        )

    def test_standard(self):
        dev = np.std(self.vector)
        self.assertTrue(
            np.array_equal(
                Normalization.standard(self.vector),
                (self.vector-np.mean(self.vector))/dev
            )
        )
