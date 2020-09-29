# -*- coding: utf-8 -*-
""" settings """

import os


###############################################################################
#                                   Datasets                                   #
###############################################################################

DATASETS_DIRECTORY = "datasets_dir"

TRAINING_DATA_DIRECTORY_DATASET_PATH = os.path.join(
    DATASETS_DIRECTORY,
    'patchcamelyon',
    'my_raw_dataset_train.json'
)
TESTING_DATA_DIRECTORY_DATASET_PATH = os.path.join(
    DATASETS_DIRECTORY,
    'patchcamelyon',
    'my_raw_dataset_test.json'
)

###############################################################################
#                                   General                                    #
###############################################################################
GENERATED_DATA_DIRECTORY = "tmp"
GENERATED_FEATS_FILENAME_TEMPLATE = 'spatial_pyramid_feats_{}_.json'
CODEBOOK_ONNX = 'codebook.onnx'

CHANNELS = 200  # number of features types
PYRAMID_LEVELS = 2
PATCH_SIZE = 16
STEP_SIZE = 8
IMAGE_WIDTH = 32  # original image with
IMAGE_HEIGHT = 32  # original image height

NUMPY_RANDOM_SEED = 0
PCA_N_COMPONENTS = 100  # set to -1 to not apply PCA
