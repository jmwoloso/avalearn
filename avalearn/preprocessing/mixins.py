#!/usr/bin/env python
"""
treatment_design.py: classes used for designing the data treatments and storing
                     the treatment plans.
"""
from abc import ABCMeta, abstractmethod
import pandas as pd



class TreatmentDesignMixin(object):
    """
    Mixin class defining core functionality for treatment design-related
    classes.
    """
    __metaclass__ = ABCMeta

    @abstractmethod

    def __init__(self, feature_columns="all", target_column=-1,
                 min_feature_significance=None):
        """

        Parameters
        ----------
        treatment_columns : one of {list of column names, list of column indices, "all"};
         default="all"; required.
            The columns that should be used during treatment design.

        target_column : one of {target column name; target column index};
         default=-1; required.
            The column name or column index where the target resides.

        min_feature_significance : one of {None, float in range(0,1); default=None; required.
            If `None`, no feature pruning will take place otherwise this is
            the minimum significance level a feature needs in order to be
            included in the final treated dataframe.

        Attributes
        ----------
        df_ : pandas.DataFrame.
            The original pandas.DataFrame used for treatment design.

        features_ : pandas.DataFrame.
            The subset of features that need treating.

        target_ : pandas.DataFrame.
            The target column.

        numeric_features_ : pandas.DataFrame.
            The numeric features in the dataframe.

        categorical_features_ : pandas.DataFrame.
            The categorical features in the dataframe.

        treatment_description_ : pandas.DataFrame-like.
            A DataFrame-like object containing information about the proposed
            treatment for the subset of features.

        treatment_plan_ : pandas.DataFrame-like.
            Augmented pandas.DataFrame constructed to accommodate lazy
            evaluation of the treatment plan after design.

        drop_features_ : Bool
            Whether to drop insignificant features or not.


        Notes
        -----

        References
        ----------

        """
        # TODO: ensure all params, attrs, etc. are documented
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.significance = min_feature_significance


    def fit(self, dataframe):
        """
        Analyze the dataframe, gather descriptive information and design the treatment.

        Parameters
        ----------
        dataframe : pandas.DataFrame instance.
            The dataframe to be used for treatment design.

        Returns
        -------
        self : object
            Returns self.
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
        self.target_ = df.ix[:, df.columns == self.target_column].columns

        # find the numeric features
        self.numeric_features_ = df.ix[:,self.features_] \
            .select_dtypes(include=['float','int']).columns

        # find categorical features
        self.categorical_features_ = df.ix[:, self.features_] \
            .select_dtypes(include=['object']).columns

        return self


    def transform(self, dataframe):
        """
        Apply the treatment to the dataframe.

        Parameters
        ----------
        dataframe : pandas.DataFrame instance; required.
            The dataframe to be transformed using the designed treatment.

        Returns
        -------
        T : pandas.DataFrame instance.
            A dataframe with treatments applied to original data.

        """
        # TODO: document all params, attrs, returns, etc.
        # TODO: may need to implement this differently for numeric versus categorical targets (i.e. not here)
        self.df_ = dataframe.copy()

        return



class LazyDataFrameMixin(pd.DataFrame):
    """
    DataFrame to record the processing steps. Can be used to apply
    transformations after the treatment is designed. Based upon the
    stackoverflow solution provided by unutbu.

    Source: https://stackoverflow.com/questions/19605537/how-to-create-lazy-evaluated-dataframe-columns-in-pandas
    """
    # TODO: implement this
    # TODO: might not need a mixin for this, could just be a straight class if no changes are needed depending on classification/regression setting
    pass


