# -*- coding: utf-8 -*-
""" spfs/core/weighting_methods """


class WeightingMethod:
    """
    Hold the options and method to assign weights based on the current spatial pyramid level

    Usage:
        WeightingMethod.get_weighting_method(WeightingMethod.KEEP_FINEST_FEATS)(current_level, levels)
    """

    KEEP_FINEST_FEATS = 0
    GENERAL_WEIGHTING = 1
    KEEP_ALL_FEATS = 2

    CHOICES = {
        KEEP_FINEST_FEATS: 'keep_finest_feats',
        GENERAL_WEIGHTING: 'general_weighting',
        KEEP_ALL_FEATS: 'keep_all_feats'
    }

    @staticmethod
    def keep_finest_feats(current_level, total_levels=None):
        """
        Calculates and returns the spatial historgram weight for the current_level keeping
        the value of the finest features. E.g.: For total_levels=2 the weights for each level are:
        0->1/4, 1->1/2, 2->1

        Args:
            current_level (int): current level to calculate
            total_levels  (int): total number of levels to apply

        Returns:
            spatial historgram weight (float)
        """
        # avoiding circular reference
        import settings

        total_levels = settings.PYRAMID_LEVELS if total_levels is None else total_levels

        assert isinstance(current_level, int)
        assert isinstance(total_levels, int)
        assert total_levels >= current_level

        return 2**(current_level - total_levels)

    @staticmethod
    def general_weighting(current_level, total_levels=None):
        """
        Calculates and returns the spatial historgram weight for the current_level its general
        weighting formula. E.g.: For total_level=2 the weights for each lever are:
        0->1/4, 1->1/4, 2->1/2

        Args:
            current_level (int): current level to calculate
            total_levels  (int): total number of levels to apply

        Returns:
            spatial historgram weight (float)
        """
        # avoiding circular reference
        import settings

        total_levels = settings.PYRAMID_LEVELS if total_levels is None else total_levels

        assert isinstance(current_level, int)
        assert isinstance(total_levels, int)
        assert total_levels >= current_level

        if current_level == 0:
            return 1/2**total_levels

        return 1/2**(total_levels - current_level + 1)

    @staticmethod
    def keep_all_feats(current_level, total_levels=None):
        """
        Does not modify the vector quantizations per level. Thus, it always returns 1. for any level.

        Args:
            current_level (int): current level to calculate
            total_levels  (int): total number of levels to apply

        Returns:
            spatial historgram weight (float)
        """
        # avoiding circular reference
        import settings

        total_levels = settings.PYRAMID_LEVELS if total_levels is None else total_levels

        assert isinstance(current_level, int)
        assert isinstance(total_levels, int)
        assert total_levels >= current_level

        return 1.

    @classmethod
    def get_weighting_method(cls, option):
        """ description """
        assert option in cls.CHOICES.keys()

        return getattr(cls, cls.CHOICES[option])
