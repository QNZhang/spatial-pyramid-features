# -*- coding: utf-8 -*-
""" core/exceptions """


class PoolingMethodInvalid(Exception):
    """
    Exception to be raised the pooling method is not defined on constants.Pooling class
    """

    def __init__(self, message=''):
        """  """
        message = 'The pooling method especified is not defined at constants.Pooling class'
        super().__init__(message)
