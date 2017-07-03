#!/usr/bin/env python
"""
base.py : Base classes for the treatment design classes.

"""
from abc import ABCMeta, abstractmethod, abstractclassmethod

import numpy as np
import pandas as pd


def _check_na_list(na_list):
    return True if len(na_list) != 0 else False

def _record_transform(transform, type, description_df):
    pass



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
                 min_feature_significance=0.05, cv=3, cv_type='loo',
                 cv_split_function=None, train_size=0.2,
                 test_size=None, random_state=None, rare_level_threshold=0.02,
                 novel_level_strategy="nan", rare_level_pooling=False,
                 make_nan_indicators=True, high_cardinality_strategy="impact"):

        # TODO: ensure all params, attrs, etc. are documented
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.significance = min_feature_significance
        self.cv = cv
        self.cv_type = cv_type
        self.split_function = cv_split_function
        if train_size is not None and test_size is None:
            self.train_size = train_size
            self.test_size = 1 - self.train_size
        elif train_size is None and test_size is not None:
            self.test_size = test_size
            self.train_size = 1 - self.test_size
        self.random_state = random_state
        self.rare_threshold = rare_level_threshold
        self.novel_strategy = novel_level_strategy
        self.make_nans = make_nan_indicators
        self.rare_pooling = rare_level_pooling
        self.high_cardinality_strategy = high_cardinality_strategy


    def fit(self, dataframe):
        """
        Create the treatment plan.
        """
        # TODO: allow for use of any sklearn.model_selection splitter classes or functions
        # TODO: if all columns are passed, we only need to transform those with NA so check for and filter only those columns
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
            self.features_ = df.loc[:, ~df.columns
                .isin([self.target_column])].columns
        else:
            self.features_ = \
                df.loc[:, df.columns
                    .isin(self.feature_columns)].columns

        # get the name of the target column in case we're passed an index value
        self.target_ = df.loc[:,
                       df.columns == self.target_column].columns

        # find the numeric features
        self.numeric_features_ = df.loc[:, self.features_] \
            .select_dtypes(include=['float', 'int']).columns

        # find categorical features
        self.categorical_features_ = df.loc[:, self.features_] \
            .select_dtypes(include=['object']).columns


        # CATEGORICAL ROUTINES
        # check for missing categorical values
        self.categorical_na_features = \
            df.loc[:, self.categorical_features_]\
                .columns[df.loc[:, self.categorical_features_]
            .isnull().any()].tolist()

        # check for empty na list
        empty_categorical = _check_na_list(self.categorical_na_features)

        if empty_categorical is False:
            print("No missing categorical values")
        else:
            print("Treating categorical features")
            # treat missing categoricals as their own level
            df.loc[:, self.categorical_na_features] = df.loc[:,
                                                     self.categorical_na_features].fillna("NaN")
            # TODO: RECORD in DescriptionDF
            # TODO: RECORD in LazyDF
            _record_transform("transform", "type", "descriptiondf")

            # we need to decide what to do with unseen levels
            # conduct statistical tests and choose best representation from:
            # 1. represent novel levels as 0 for each previously seen level
            # indicator
            # 2. represent novel levels as uncertainty among rare levels (using
            # self.rare_threshold)
            # 3. represent novel levels proportional to known levels
            # 4. created pooled novel indicator

            # create indicator variables for categorical features and clone
            # dataframe for each of the categorical encodings above
            # TODO: Refactor this since it could result in huge memory usage
            df1 = df.copy()
            df2 = df.copy()
            df3 = df.copy()
            df4 = df.copy()

            # create impact codings
            impact_dict = dict()
            for column in self.categorical_features_:
                for category in df.loc[:, column].unique().tolist():
                    impact_dict[column] = dict()
                    # df[df.loc[:, self.target_]][df.loc[:, column] == category].mean()
                    impact_dict[column][category] = \
                        df.loc[:, self.target_]\
                        [df.loc[:, self.target_] == 1]\
                        [df.loc[:, column] == category].mean() - df.loc[:,
                                                                 self.target_].mean()
                impact_column = category + "_impact"
                val_dict = dict()
                for idx,row in df.loc[:, category].iteritems():
                    c = df.loc[:, category] == row
                    # TODO: might not go in the mixin since it is classification specific
                    d = df.loc[:, self.target_] == 1
                    df.loc[idx, impact_column] = df.loc[:]

            df1 = pd.get_dummies(df,
                                 prefix=None,
                                 prefix_sep="_",
                                 dummy_na=False,
                                 columns=self.categorical_features_)
            # TODO: Implement #2
            # TODO: Implement #3

            df4 = pd.get_dummies(df,
                                 prefix=None,
                                 prefix_sep="_",
                                 dummy_na=True,
                                 columns=self.categorical_features_)








        # NUMERIC ROUTINES
        self.numeric_na_features = \
            df.loc[:, self.numeric_features_]\
                .columns[df.loc[:, self.numeric_features_]
            .isnull().any()].tolist()

        # TODO: check for an empty list
        empty_numeric = _check_na_list(self.numeric_na_features)

        # we're done with this if there are none
        if empty_numeric is False:
            print("No missing numeric values.")

        else:

            # replace missing numeric values in the dataframe
            # TODO: RECORD in DescriptionDF
            # TODO: RECORD in LazyDF
            df.loc[:, self.numeric_na_features] = \
                df.loc[:, self.numeric_na_features]\
                    .fillna(df.loc[:, self.numeric_na_features].mean())

            _record_transform("transform", "type", "descriptiondf")

        return self

    def transform(self, dataframe):
        """
        Apply the treatment to the dataframe.
        """
        # TODO: document all params, attrs, returns, etc.
        # TODO: may need to implement this differently for numeric versus categorical targets (i.e. not here)
        self.df_ = dataframe.copy()

        # CATEGORICAL ROUTINES
        # we need to decide what to do with unseen levels
        # conduct statistical tests and choose best representation from:
        # 1. represent novel levels as 0 for each previously seen level
        # indicator
        # 2. represent novel levels as uncertainty among rare levels (using
        # self.rare_threshold)
        # 3. represent novel levels proportional to known levels
        # 4. created pooled novel indicator

        return


    def _impact_coding(self):
        def logit(p):
            return np.log(p / 1 - p)

        # calculate the impact codes





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