# -*- coding: utf-8 -*-
""" main """

import numpy as np

import settings
from core.feature_extractors import SpatialPyramidFeatures
from utils.datasets.handlers import InMemoryDBHandler, FeatsHandler, LazyDBHandler


np.random.seed(settings.NUMPY_RANDOM_SEED)


def main():
    pass


if __name__ == '__main__':
    main()

    # FeatsHandler().create_subsets(20, True)
    # train_feats, train_labels, test_feats, test_labels = FeatsHandler(percentage=20, verbose=True)()

    spf = SpatialPyramidFeatures(InMemoryDBHandler)
    # spf = SpatialPyramidFeatures(LazyDBHandler)
    # func:create_codebook processed in 5602.9401 seconds
    # func:create_codebook processed in 5549.5434 seconds
    spf.create_codebook()
    # old time: func:create_spatial_pyramid_features processed in 93547.0923 seconds
    # new time: func:create_spatial_pyramid_features processed in  7969.3614 seconds
    # spf.create_spatial_pyramid_features()
