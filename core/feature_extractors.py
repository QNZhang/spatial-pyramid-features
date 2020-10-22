# -*- coding: utf-8 -*-
""" core/feature_extractors """

import os
import json
from random import sample
import warnings

import cv2 as cv
# import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as rt
from gutils.context_managers import tqdm_joblib
from gutils.decorators import timing
from gutils.image_processing import get_patches
from gutils.numpy_.numpy_ import get_unique_rows
# from gutils.numpy_.images import ZeroPadding
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from skl2onnx import to_onnx
from tqdm import tqdm

import settings
from constants import Pooling
from core.exceptions import PoolingMethodInvalid, WrongSpatialPyramidSubregionsNumber
from core.weighting_methods import WeightingMethod
from utils.datasets.templates import DatasetItemsTemplate
from utils.descriptors import get_sift_descriptors
from utils.utils import get_uint8_image, apply_pca, using_quick_tests


class SpatialPyramidFeatures:
    """
    Holds serveral method to create the codebook, extract the spatial pyramid features/histograms
    and save it in JSON files

    Usage:
        from utils.datasets.handlers import InMemoryDBHandler, LazyDBHandler

        spf = SpatialPyramidFeatures(InMemoryDBHandler)  # LazyDBHandler is another option
        spf.create_codebook()
        spf.create_spatial_pyramid_features()
    """

    def __init__(self, db_handler_class):
        """ Verifies the consistency in settings values and initializes the instance """
        assert settings.PATCH_SIZE % 2**settings.PYRAMID_LEVELS == 0 or settings.PATCH_SIZE <= 0
        if using_quick_tests():
            assert settings.PCA_N_COMPONENTS < settings.QUICK_TESTS

        self.db_handler_class = db_handler_class

    def get_concatenated_weighted_histogram(
            self, img, pyramid_levels=settings.PYRAMID_LEVELS,
            keypoint_step_size=settings.KEYPOINTS_STEP_SIZE
    ):
        """
        Process an image considering the pyramid level specified and returns its long
        contatenated-weighted histogram using the SIFT descriptors

        Args:
            img     (np.ndarray): image loaded using numpy
            pyramid_levels (int): number of pyramid levels to be used

        Returns:
            histogram np.array with length = round(channels * (1/3) * (4**(pyramid_levels+1)-1))
        """
        assert isinstance(img, np.ndarray)
        assert isinstance(pyramid_levels, int)
        assert min(img.shape[:2]) > 2**pyramid_levels, \
            "the image dimensions {} must be bigger than 2**pyramid_levels"\
            .format(str(img.shape[:2]))

        histogram = []
        codebook = self.load_codebook()
        input_name = codebook.get_inputs()[0].name
        label_name = codebook.get_outputs()[0].name

        for level in range(pyramid_levels+1):
            weight = WeightingMethod.get_weighting_method(
                settings.PYRAMID_FEATURE_WEIGHTING_METHOD)(level, pyramid_levels)
            patches = list(get_patches(get_uint8_image(img), img.shape[1]//2**level, img.shape[0]//2**level))

            if len(patches) != 2**(2*level):
                raise WrongSpatialPyramidSubregionsNumber(2**(2*level), len(patches))

            for patch in patches:
                des = get_sift_descriptors(patch, keypoint_step_size)

                if des.shape[0] != 0:
                    des = des if des.dtype == np.float32 else des.astype(np.float32)
                    # ENCODED DESCRIPTORS POOLING OPERATIONS ##################
                    # NOTE: Because we're using kmeans to assing each SIFT descriptor
                    # to a class/cluster, all the encoded descriptors will be
                    # 1-D zero vectors with only one of their elements set to 1. e.g.:
                    # [0, 1, 0, 0, 0]
                    # Thus, SUM pooling will return a vector quantization (histogram)
                    # and MAX pooling will return a vector of ones and zeros
                    if settings.ENCODED_DESCRIPTORS_POOLING_METHOD in (Pooling.SUM, Pooling.MAX):
                        predictions = codebook.run([label_name], {input_name: des})[0]
                        patch_histogram = np.bincount(predictions, minlength=settings.CHANNELS)\
                                            .reshape(1, -1).ravel()

                        if settings.ENCODED_DESCRIPTORS_POOLING_METHOD == Pooling.MAX:
                            patch_histogram[np.nonzero(patch_histogram)] = 1

                        histogram.append(weight*patch_histogram)
                    else:
                        raise PoolingMethodInvalid

        return np.concatenate(histogram)

    def process_image(self, img, pyramid_levels=settings.PYRAMID_LEVELS,
                      keypoint_step_size=settings.KEYPOINTS_STEP_SIZE):
        """
        Process an image using the pyramid_levels especified, and returns its normalized
        spatial pyramid features

        Args:
            img         (np.ndarray): image loaded using numpy
            pyramid_levels     (int): number of pyramid levels to be used
            keypoint_step_size (int): step size & keypoint diameter

        Returns:
            [np.ndarray] with length = round(channels * (1/3) * (4**(pyramid_levels+1)-1))]
        """
        assert isinstance(img, np.ndarray)
        assert isinstance(pyramid_levels, int)

        histogram = self.get_concatenated_weighted_histogram(
            settings.TWEAK_IMAGE_METHOD(img, pyramid_levels), pyramid_levels, keypoint_step_size)

        if np.count_nonzero(histogram) == 0:
            warnings.warn('An image without feature descriptors was processed', UserWarning)
            return [histogram]

        return [settings.NORMALIZATION(histogram)]

    @timing
    def create_codebook(
            self,
            patches_percentage=settings.CODEBOOK_PATCHES_PERCENTAGE,
            patch_size=settings.PATCH_SIZE,
            step_size=settings.PATCH_STEP_SIZE,
            keypoint_step_size=settings.CODEBOOK_CREATION_KEYPOINTS_STEP_SIZE,
            save=True
    ):
        """
        Trains a Kmeans classifier/codebook using a percentage of the trainig featues and
        saves it if save=True

        Args:
            patches_percentage  (float): percentage of patches to be used to build the codebook
            patch_size            (int): size of the patch
            step_size             (int): patch stride
            keypoint_step_size    (int): step size & keypoint diameter
            save                 (bool): persist the model in a ONNX file

        Returns:
            codebook (Kmeans classifier)
        """
        assert isinstance(patches_percentage, float)
        assert 0 < patches_percentage <= 1

        training_feats = self.db_handler_class(True)()[0]

        def process_column(patch_size, step_size, img):
            all_descriptors = []

            if patch_size > 0:
                patches = [i for i in get_patches(
                    img, patch_size, patch_overlapping=patch_size-step_size)]

                for patch in sample(patches, round(len(patches) * patches_percentage)):
                    descriptors = get_sift_descriptors(patch, keypoint_step_size)

                    if descriptors.any():
                        all_descriptors.append(descriptors)

            else:
                all_descriptors = [get_sift_descriptors(img, keypoint_step_size)]

            if all_descriptors:
                return np.concatenate(all_descriptors)

            return np.empty([0, 128], dtype=np.float32)

        with tqdm_joblib(tqdm(desc="Processing images", total=training_feats.num_samples)):
            descriptors_list = Parallel(n_jobs=-1)(delayed(process_column)(
                patch_size, step_size, training_feats.get_sample(col)) for col in range(training_feats.num_samples))

        if not descriptors_list:
            warnings.warn(
                'No descriptors were found; thus, the codebook could not be created', UserWarning)
            return None

        selected_descriptors = np.concatenate(descriptors_list)
        print("{} descriptors found".format(selected_descriptors.shape[0]))
        selected_descriptors = get_unique_rows(selected_descriptors)
        print("{} unique descriptors".format(selected_descriptors.shape[0]))

        print("Training KMeans classifer...")
        kmeans = KMeans(n_clusters=settings.CHANNELS, random_state=settings.RANDOM_STATE,
                        verbose=settings.KMEANS_VERBOSE).fit(selected_descriptors)

        if save:
            print("Saving codebook/Kmeans classifier")
            onx = to_onnx(kmeans, selected_descriptors[:1].astype(np.float32))

            if not os.path.isdir(settings.GENERATED_DATA_DIRECTORY):
                os.mkdir(settings.GENERATED_DATA_DIRECTORY)

            with open(os.path.join(settings.GENERATED_DATA_DIRECTORY, settings.CODEBOOK_ONNX),
                      'wb') as file_:
                file_.write(onx.SerializeToString())

        return kmeans

    @staticmethod
    def load_codebook():
        """ Loads the ONNX file and returns the codebook """
        return rt.InferenceSession(
            os.path.join(settings.GENERATED_DATA_DIRECTORY, settings.CODEBOOK_ONNX))

    def __get_histograms(self, dataset):
        """
        Gets the histograms/features from the dataset and returns them

        Args:
            dataset (DatasetItemsTemplate): Dataset

        Returns:
            np.ndarray [samples, features]
        """
        assert isinstance(dataset, DatasetItemsTemplate)

        with tqdm_joblib(tqdm(total=dataset.num_samples)):
            db_histograms = Parallel(n_jobs=-1)(delayed(self.process_image)(
                dataset.get_sample(col)) for col in range(dataset.num_samples))

        db_histograms = np.concatenate(db_histograms)

        return db_histograms

    @timing
    def create_spatial_pyramid_features(self):
        """
        * Processes the dataset
        * Calculates all the histograms
        * Optionaly applies PCA to the histograms/feature vectors
        * Saves the processed dataset
        """
        train_feats, train_labels, test_feats, test_labels = self.db_handler_class(True)()

        print("Getting histograms from training dataset")
        train_histograms = self.__get_histograms(train_feats)
        print("Getting histograms from testing dataset")
        test_histograms = self.__get_histograms(test_feats)

        if settings.PCA_N_COMPONENTS != -1:
            # TODO: maybe I shouldn't apply PCA to train and test separately....
            print("Applying PCA to training spatial pyramid features")
            train_histograms = apply_pca(train_histograms)
            print("Applying PCA to testing spatial pyramid features")
            test_histograms = apply_pca(test_histograms)

        for db_split, feats, labels in [
                ('train', train_histograms, train_labels), ('test', test_histograms, test_labels)]:
            formatted_data = dict(
                codes=feats.T.tolist(),
                labels=labels.tolist()
            )
            saving_path = os.path.join(
                settings.GENERATED_DATA_DIRECTORY,
                settings.GENERATED_FEATS_FILENAME_TEMPLATE.format(db_split)
            )

            with open(saving_path, 'w') as file_:
                json.dump(formatted_data, file_)
