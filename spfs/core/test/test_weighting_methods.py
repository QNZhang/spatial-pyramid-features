# -*- coding: utf-8 -*-
""" spfs/core/test_weighting_methods """

import unittest

from spfs.core.weighting_methods import WeightingMethod


class TestWeightingMethod(unittest.TestCase):

    def test_keep_finest_feats(self):
        self.assertEqual(WeightingMethod.keep_finest_feats(0, 2), 0.25)
        self.assertEqual(WeightingMethod.keep_finest_feats(1, 2), 0.5)
        self.assertEqual(WeightingMethod.keep_finest_feats(2, 2), 1)

    def test_general_weighting(self):
        self.assertEqual(WeightingMethod.general_weighting(0, 2), 0.25)
        self.assertEqual(WeightingMethod.general_weighting(1, 2), 0.25)
        self.assertEqual(WeightingMethod.general_weighting(2, 2), 0.5)

    def test_keep_all_feats(self):
        self.assertEqual(WeightingMethod.keep_all_feats(0, 2), 1.)
        self.assertEqual(WeightingMethod.keep_all_feats(1, 2), 1.)
        self.assertEqual(WeightingMethod.keep_all_feats(2, 2), 1.)

    def test_get_weighting_method(self):
        self.assertEqual(
            WeightingMethod.get_weighting_method(WeightingMethod.KEEP_FINEST_FEATS).__name__,
            WeightingMethod.CHOICES[WeightingMethod.KEEP_FINEST_FEATS]
        )
        self.assertEqual(
            WeightingMethod.get_weighting_method(WeightingMethod.GENERAL_WEIGHTING).__name__,
            WeightingMethod.CHOICES[WeightingMethod.GENERAL_WEIGHTING]
        )
        self.assertEqual(
            WeightingMethod.get_weighting_method(WeightingMethod.KEEP_ALL_FEATS).__name__,
            WeightingMethod.CHOICES[WeightingMethod.KEEP_ALL_FEATS]
        )


if __name__ == '__main__':
    unittest.main()
