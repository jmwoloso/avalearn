#!/usr/bin/env python
"""
base.py : Base classes for the treatment design classes.

"""
from abc import ABCMeta, abstractmethod

import pandas as pd


class BaseTreatmentDesign(object):
    """
    Base class for Treatment Design classes.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, dataframe):
        """Defines the fit routines."""
        pass

    @abstractmethod
    def transform(self, dataframe):
        """Defines the transformation routines."""
        pass

    @abstractmethod
    def fit_transform(self, dataframe):
        """Defines fit_transform routines."""
        pass



class DelayedDataFrame(pd.DataFrame):
    """
    DataFrame to record the processing steps. Can be used to apply
    transformations after the treatment is designed. Based upon the
    stackoverflow solution provided by unutbu.

    Source: https://stackoverflow.com/questions/19605537/how-to-create-lazy-evaluated-dataframe-columns-in-pandas
    """
    # TODO: implement this
    # TODO: might not need a mixin for this, could just be a straight class if no changes are needed depending on classification/regression setting
    pass



class TreatmentDescriptionDataFrame(pd.DataFrame):
    def __init__(self):
        # columns for the treament plan description df
        columns = ["new_variable_name", "new_variable_significance",
                   "extra_degrees_of_freedom", "original_variable",
                   "transformation_type"]
        super().__init__(columns=columns)