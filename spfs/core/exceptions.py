# -*- coding: utf-8 -*-
""" spfs/core/exceptions """


class PoolingMethodInvalid(Exception):
    """
    Exception to be raised the pooling method is not defined on constants.Pooling class
    """
    message = 'The pooling method especified is not defined at constants.Pooling class'

    def __init__(self):
        """  """
        super().__init__(self.message)


class WrongSpatialPyramidSubregionsNumber(Exception):
    """
    Exception to be raised the spatial pyramid subregions generated at a certain levels is not
    correct. It must always be 2**(d*current_level) where the is the dimensions of the matrix
    This application only works with 2D matrices; consequently, in this program d = 2
    """
    message = 'Wrong number of Spatial Pyramid Subregions. You got {} but it sholud be {}'

    def __init__(self, target_num, obtained_num):
        """
        Args:
            target_num   (int): expected number of subregions
            obtained_num (int): obtained number of subregions
        """
        super().__init__(self.message.format(target_num, obtained_num))
