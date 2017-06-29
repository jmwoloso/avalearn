#!/usr/bin/env python
"""
data_prep.py: classes for designing data treatment.
"""
import pandas as pd

from .base import TreatmentDesignMixin



class ClassificationTreatmentDesign(TreatmentDesignMixin):
    """
    Class for designing treatments for classification tasks.
    """
    def __init__(self, feature_columns="all", target_column=-1,
                 min_feature_significance=None):
        super().__init__(feature_columns, target_column, min_feature_significance)

    def fit(self, dataframe):
        super().fit(dataframe)



class RegressionTreatmentDesign(TreatmentDesignMixin):
    """
    Class for designing treatments for regression tasks.
    """
    def __init__(self, feature_columns="all", target_column=-1,
                 min_feature_significance=None):
        super().__init__(feature_columns, target_column, min_feature_significance)

    def fit(self, dataframe):
        super().fit(dataframe)