# -*- coding: utf-8 -*-
""" main """

import numpy as np

import settings
from core.feature_extractors import SpatialPyramidFeatures
from utils.datasets.patchcamelyon import DBhandler


np.random.seed(settings.NUMPY_RANDOM_SEED)


def main():
    pass


if __name__ == '__main__':
    main()

    spf = SpatialPyramidFeatures(DBhandler)
    spf.create_codebook()
    spf.create_spatial_pyramid_features()
