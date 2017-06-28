#!/usr/bin/env python
"""
data_prep.py: classes for designing data treatment.
"""
import pandas as pd

from .mixins import TreatmentDesignMixin

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



class TreatmentDescriptionDF(pd.DataFrame):
    """
    pd.DataFrame that holds descriptive information about the proposed
    treatment.
    """
    def __init__(self):
        # columns for the treament plan description df
        columns = ["new_variable_name", "new_variable_significance",
                   "extra_degrees_of_freedom", "original_variable",
                   "transformation_type"]
        super().__init__(columns=columns)