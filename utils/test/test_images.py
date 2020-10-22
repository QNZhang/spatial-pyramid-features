# -*- coding: utf-8 -*-
""" utils/test/test_images """

import unittest

import numpy as np

from utils.images import TweakImage


class TestTweakImage(unittest.TestCase):

    def setUp(self):
        self.image = np.full([18, 18], 5)

    def test_get_zero_padded_image(self):
        self.assertEqual(
            TweakImage.get_zero_padded_image(self.image, 2).shape,
            (20, 20)
        )

    def test_get_adjusted_image(self):
        self.assertEqual(
            TweakImage.get_adjusted_image(self.image, 2).shape,
            (16, 16)
        )


if __name__ == '__main__':
    unittest.main()
