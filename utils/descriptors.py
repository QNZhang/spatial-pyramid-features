# -*- coding: utf-8 -*-
""" utils/descriptors """

import cv2 as cv
import numpy as np

from gutils.image_processing import get_patches

import settings
from utils.utils import get_uint8_image


def get_sift_descriptors(img, pyramid_levels=settings.PYRAMID_LEVELS):
    """
    Process an image using the especified patch_size and step_size; and returns its SIFT
    descriptors

    Args:
        img     (np.ndarray): image loaded using numpy
        pyramid_levels (int): number of pyramid levels to be used
        patch_size     (int): size of the patchw
        step_size      (int): stride

    Returns:
        np.array [num of key points, 128-SIFT descriptors]
    """
    assert isinstance(img, np.ndarray)
    assert isinstance(pyramid_levels, int)
    assert min(img.shape[:2]) > 2**pyramid_levels, \
        "the image dimensions {} must be bigger than 2**pyramid_levels"\
        .format(str(img.shape[:2]))

    sift = cv.SIFT_create()
    descriptors = np.empty([0, 128], dtype=np.float32)

    for level in range(pyramid_levels+1):
        for patch in get_patches(get_uint8_image(img), min(img.shape[:2])//2**level):
            kp, des = sift.detectAndCompute(patch, None)
            if des is not None:
                # print('{} descriptors at level {}'.format(des.shape[0], level))
                descriptors = np.r_[descriptors, des]
                # patch = cv.drawKeypoints(patch, kp, patch, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                # plt.imshow(patch)
                # plt.show()

    return descriptors
