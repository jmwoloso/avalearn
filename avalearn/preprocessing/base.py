#!/usr/bin/env python
"""
base.py : Base classes for the treatment design classes.

"""
from abc import ABCMeta, abstractmethod, abstractclassmethod

import pandas as pd



class BaseTreatmentDesign(object):
    """
    Base class for Treatment Design classes.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, dataframe):
        pass

    # @abstractclassmethod
    # @abstractmethod
    # def fit_transform(self, dataframe):
    #     pass

    # @abstractclassmethod
    @abstractmethod
    def transform(self, dataframe):
        pass



class TreatmentDesignMixin(BaseTreatmentDesign):
    """
    Mixin class defining core functionality for treatment design-related
    classes.
    """

    def __init__(self, feature_columns="all", target_column=-1,
                 min_feature_significance=None):

        # TODO: ensure all params, attrs, etc. are documented
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.significance = min_feature_significance


    def fit(self, dataframe):
        """
        Create the treatment plan.
        """
        # TODO: add versionadded markers
        # TODO: create base transformer that has basic functionality
        # TODO: pretty print signature after instantiation (in base transformer)
        # ensure we were passed a dataframe
        # NOTE: doesn't check that any 'object' columns might actually be
        # numeric
        # TODO: check 'object' columns to see if they're really numeric
        # _check_dframe(dataframe)

        # TODO: need to check whether the list contains numbers and
        # check valid kwarg values were passed
        # _check_treatment_kwargs(self.treatment_columns,
        #                         self.target_column)

        # TODO: check whether significance is a valid value first
        self.drop_features_ = False if self.significance is None else True

        # instantiate the TreatmentDescription DF
        self.treatment_description_ = TreatmentDescriptionDF()

        # instantiate the LazyDataFrame that will perform the transformations
        self.treatment_plan_ = LazyDataFrame()

        # TODO: do we really need to store this?
        # self.df_ = dataframe.copy()
        df = dataframe

        # solidify feature columns
        if self.feature_columns == "all":
            self.features_ = df.ix[:, ~df.columns
                .isin(self.target_column)].columns
        else:
            self.features_ = \
                df.ix[:, df.columns
                    .isin(self.feature_columns)].columns

        # get the name of the target column in case we're passed an index value
        self.target_ = df.ix[:,
                       df.columns == self.target_column].columns

        # find the numeric features
        self.numeric_features_ = df.ix[:, self.features_] \
            .select_dtypes(include=['float', 'int']).columns

        # find categorical features
        self.categorical_features_ = df.ix[:, self.features_] \
            .select_dtypes(include=['object']).columns

        return self

    def transform(self, dataframe):
        """
        Apply the treatment to the dataframe.
        """
        # TODO: document all params, attrs, returns, etc.
        # TODO: may need to implement this differently for numeric versus categorical targets (i.e. not here)
        self.df_ = dataframe.copy()

        return



class LazyDataFrame(pd.DataFrame):
    """
    DataFrame to record the processing steps. Can be used to apply
    transformations after the treatment is designed. Based upon the
    stackoverflow solution provided by unutbu.

    Source: https://stackoverflow.com/questions/19605537/how-to-create-lazy-evaluated-dataframe-columns-in-pandas
    """
    # TODO: implement this
    # TODO: might not need a mixin for this, could just be a straight class if no changes are needed depending on classification/regression setting
    pass



class TreatmentDescriptionDF(pd.DataFrame):
    def __init__(self):
        # columns for the treament plan description df
        columns = ["new_variable_name", "new_variable_significance",
                   "extra_degrees_of_freedom", "original_variable",
                   "transformation_type"]
        super().__init__(columns=columns)