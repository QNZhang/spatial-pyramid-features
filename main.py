# -*- coding: utf-8 -*-
""" main """

import numpy as np

import settings
from spfs.core.feature_extractors import SpatialPyramidFeatures
from spfs.utils.datasets.handlers import InMemoryDBHandler, LazyDBHandler
from spfs.utils.utils import FeaturesEvaluator


np.random.seed(settings.NUMPY_RANDOM_SEED)


def main():
    pass


if __name__ == '__main__':
    main()

    # spf = SpatialPyramidFeatures(InMemoryDBHandler)
    spf = SpatialPyramidFeatures(LazyDBHandler)
    # func:create_codebook processed in 5602.9401 seconds
    # func:create_codebook processed in 5549.5434 seconds
    # func:create_codebook processed in 17794.9059 seconds
    spf.create_codebook()
    # old time: func:create_spatial_pyramid_features processed in 93547.0923 seconds
    # new time: func:create_spatial_pyramid_features processed in  7969.3614 seconds
    spf.create_spatial_pyramid_features()

    ###########################################################################
    #                        Evaluation with linear SVC                       #
    ###########################################################################
    FeaturesEvaluator.apply_linear_svc()
