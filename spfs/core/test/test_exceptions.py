# -*- coding: utf-8 -*-
""" spfs/core/test/test_exceptions """

import unittest

from spfs.core.exceptions import PoolingMethodInvalid, WrongSpatialPyramidSubregionsNumber


class TestPoolingMethodInvalid(unittest.TestCase):

    def test_message(self):
        with self.assertRaisesRegex(PoolingMethodInvalid, PoolingMethodInvalid.message):
            raise PoolingMethodInvalid()


class TestWrongSpatialPyramidSubregionsNumber(unittest.TestCase):

    def test_message(self):
        message = WrongSpatialPyramidSubregionsNumber.message.format(4, 5)

        with self.assertRaisesRegex(WrongSpatialPyramidSubregionsNumber, message):
            raise WrongSpatialPyramidSubregionsNumber(4, 5)


if __name__ == '__main__':
    unittest.main()
