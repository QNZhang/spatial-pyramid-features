# -*- coding: utf-8 -*-
""" utils/descriptors """

import cv2 as cv
import numpy as np

import settings


def get_sift_descriptors(img, keypoint_step_size=settings.KEYPOINTS_STEP_SIZE):
    """
    Gets the SIFT descriptors using the specified keypoint_step_size and returns them

    Args:
        img         (np.ndarray): image loaded using numpy
        keypoint_step_size (int): step size & keypoint diameter

    Returns:
        np.array with shape (num of key points, 128-SIFT descriptors)
    """
    assert isinstance(img, np.ndarray)
    assert min(img.shape[:2]) >= keypoint_step_size, \
        "keypoint_step_size must be greater or equal to the width or height of the provided img"

    sift = cv.SIFT_create()
    keypoints = [cv.KeyPoint(x, y, keypoint_step_size)
                 for y in range(0, img.shape[0], keypoint_step_size)
                 for x in range(0, img.shape[1], keypoint_step_size)]
    kp, des = sift.compute(img, keypoints)
    # kp, des = sift.detectAndCompute(img, None)
    # kp_img = cv.drawKeypoints(img, kp, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # plt.imshow(kp_img)
    # plt.show()

    if des is not None:
        return des

    return np.empty([0, 128], dtype=np.float32)
