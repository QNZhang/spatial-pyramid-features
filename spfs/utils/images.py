# -*- coding: utf-8 -*-
""" spfs/utils/images """

from gutils.numpy_.images import ZeroPadding


class TweakImage:
    """
    Holds methods to adjust an image and make it compatible witht the spatial pyramid analysis
    """

    @staticmethod
    def get_zero_padded_image(img, pyramid_levels=None):
        """
        Returns a zero padded image such each axis is a multiple of pyramid_levels

        Args:
            img         (np.ndarray): image loaded using numpy
            pyramid_levels     (int): number of pyramid levels to be used

        Returns:
            np.ndarray
        """
        # avoiding circular reference
        import settings  # pylint: disable=import-outside-toplevel; # noqa

        if pyramid_levels is None:
            pyramid_levels = settings.PYRAMID_LEVELS

        row_padding = col_padding = 0
        max_cell_number = 2**pyramid_levels
        row_modulo = img.shape[0] % max_cell_number
        col_modulo = img.shape[1] % max_cell_number

        if row_modulo != 0:
            row_padding = max_cell_number - row_modulo

        if col_modulo != 0:
            col_padding = max_cell_number - col_modulo

        return ZeroPadding(img, img.shape[0] + row_padding, img.shape[1] + col_padding)()

    @staticmethod
    def get_adjusted_image(img, pyramid_levels=None):
        """
        Returns a modified image such each axis is a multiple of pyramid_levels.
        This method removes the last cols/rows to achieve the desired result.

        Args:
            img         (np.ndarray): image loaded using numpy
            pyramid_levels     (int): number of pyramid levels to be used

        Returns:
            np.ndarray
        """
        # avoiding circular reference
        import settings  # pylint: disable=import-outside-toplevel; # noqa

        if pyramid_levels is None:
            pyramid_levels = settings.PYRAMID_LEVELS

        max_cell_number = 2**pyramid_levels
        row_modulo = img.shape[0] % max_cell_number
        col_modulo = img.shape[1] % max_cell_number

        return img[0:img.shape[0] - row_modulo, 0:img.shape[1] - col_modulo]
