#!/usr/bin/env python
"""
base.py : Base classes for the treatment design classes.

"""
import pandas as pd


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