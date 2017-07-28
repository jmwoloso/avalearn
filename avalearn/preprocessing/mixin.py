#!/usr/bin/env python
import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
from .base import TreatmentDescriptionDataFrame, \
    DelayedDataFrame
from ..utils._validation import _check_dframe, _check_feature_target, \
    _check_train_test_size, _check_column_dtypes, \
    _check_boolean, \
    _check_int_keyword, _check_float_keyword, _check_str_keyword, \
    _check_min_feature_significance, _check_hc_threshold, \
    _check_ordinal_mapping, _check_cat_fill_value


class TreatmentDesignMixin(object):
    """
    Mixin class defining core functionality for treatment design-related
    classes.
    """

    def __init__(self, features="all", target=-1,
                 min_feature_significance=None, cv=3, cv_type='loo',
                 cv_split_function=None, train_size=0.2,
                 test_size=None, random_state=None,
                 unique_level_min_percent=0.02,
                 novel_level_strategy="nan", rare_level_pooling=False,
                 make_nan_indicators=True,
                 high_cardinality_strategy="impact",
                 downstream_context=None, remove_duplicates=False,
                 feature_scaling=False,
                 rare_level_significance=None,
                 feature_engineering=True,
                 convert_dtypes=True,
                 find_ordinals=True,
                 ordinals=None,
                 high_cardinality_threshold=None,
                 n_jobs=1,
                 ordinal_mapping=None,
                 categorical_fill_value="NaN",
                 ordinal_fill_value=-1):

        # TODO: ensure all params, attrs, etc. are documented
        self.features = features
        self.target = target
        self.significance = min_feature_significance
        self.cv = cv
        self.cv_type = cv_type
        self.split_function = cv_split_function
        self.train_size = train_size
        self.test_size = test_size
        self.random_state = random_state
        self.unique_percent = unique_level_min_percent
        self.novel_strategy = novel_level_strategy
        self.make_nans = make_nan_indicators
        self.rare_pooling = rare_level_pooling
        self.high_cardinality_strategy = high_cardinality_strategy
        self.high_cardinality_threshold = high_cardinality_threshold
        self.context = downstream_context
        self.deduplicate = remove_duplicates
        self.scaling = feature_scaling
        self.rare_significance = rare_level_significance
        self.feature_engineering = feature_engineering
        self.convert_dtypes = convert_dtypes
        self.find_ordinals = find_ordinals
        self.ordinals=ordinals
        self.n_jobs = n_jobs
        self.mapping = ordinal_mapping
        self.categorical_fill_value = categorical_fill_value
        self.ordinal_fill_value = ordinal_fill_value

    def fit(self, dataframe):
        """
        Create the treatment plan.
        """
        # input validation
        self._validate_params(dataframe)
        
        # with input validated, create the TreatmentDescriptionDF object
        self.TreatmentDescriptionDF_ = TreatmentDescriptionDataFrame()
        self.DelayedDF_ = DelayedDataFrame()
        
        # copy the dataframe for modification
        self.df_ = dataframe.copy()
        
        # get the indices for nan columns
        self._get_nan_indices()

        # fill in missing values
        self._fill_na()

        # self._get_categorical_value_counts()
        # make indicators for additional modeling and significance testing
        self._make_indicators()

        # make indicators for whether a value replacement occurred
        self._make_replacement_indicators(dataframe)
        
        return self
    
    def transform(self, dataframe):
        """
        Apply the treatment to the dataframe.
        """
        pass

    def fit_transform(self, dataframe):
        """
        Create and apply the treatment plan.
        """
        pass

    def _validate_params(self, dataframe):
        """
        Validates the parameters passed in during initialization.
        """
        _check_dframe(dataframe=dataframe)
    
        _check_min_feature_significance(self.significance,
                                        ["float between 0.0 and 1.0",
                                         "1/n_features",
                                         None])
    
        # TODO: add enhancement to allow value to be set based upon the dataset instead of fixed int values
        _check_hc_threshold(self.high_cardinality_threshold,
                            ["int >= 0", None])
    
        _check_ordinal_mapping(self.mapping,
                               [dict, None],
                               dataframe.columns.tolist())
    
        _check_cat_fill_value(self.categorical_fill_value,
                              "categorical_fill_value")
    
        _check_float_keyword(self.unique_percent,
                             "unique_level_min_percent",
                             [0, 1])
    
        _check_int_keyword(self.n_jobs,
                           "n_jobs",
                           [-1, np.inf])
    
        _check_int_keyword(self.ordinal_fill_value,
                           "ordinal_fill_value",
                           None)
    
        _check_str_keyword(self.context,
                           "downstream_context",
                           ["pipeline", None])
    
        _check_str_keyword(self.high_cardinality_strategy,
                           "high_cardinality_strategy",
                           ["impact", "indicators"])
    
        _check_str_keyword(self.novel_strategy,
                           "novel_level_strategy",
                           ["known", "zero", "nan", "rare", "pooled"])
    
        _check_boolean(self.rare_pooling, "rare_level_pooling")
        _check_boolean(self.make_nans, "make_nan_indicators")
        _check_boolean(self.find_ordinals, "find_ordinals")
        _check_boolean(self.deduplicate, "remove_duplicates")
        _check_boolean(self.scaling, "feature_scaling")
        _check_boolean(self.feature_engineering, "feature_engineering")
    
        self.remove_features_ = False if self.significance is None else True
    
        self.features_, self.target_ = _check_feature_target(
            df_columns=dataframe.columns,
            features=self.features,
            target=self.target)
    
        self.train_size_ = _check_train_test_size(self.train_size,
                                                  self.test_size)
    
        self.numeric_, self.categorical_, self.ordinal_, self.mapping_, \
        self.numeric_has_nan_, self.categorical_has_nan_, self.ordinal_has_nan_ = \
            _check_column_dtypes(dataframe=dataframe,
                                 features=self.features_,
                                 target=self.target_,
                                 convert_dtypes=self.convert_dtypes,
                                 ordinals=self.ordinals,
                                 find_ordinals=self.find_ordinals,
                                 categorical_fill_value=self.categorical_fill_value)
    
        self.random_state_ = check_random_state(self.random_state)
    
        # TODO: finish validation routines
        # TODO: check cv, cv_type, cv_split_function, rare_level_significance, convert_dtypes, ordinals

    def _get_nan_indices(self):
        # get the indices of missing values
        self.nan_indices_ = dict()
    
        self.nan_indices_["numeric_has_nan_"] = dict()
        self.nan_indices_["categorical_has_nan_"] = dict()
        self.nan_indices_["ordinal_has_nan_"] = dict()
    
        for column in self.numeric_has_nan_:
            self.nan_indices_["numeric_has_nan_"][column] = \
                np.where(self.df_.loc[:, column].isnull())[0]
    
        for column in self.categorical_has_nan_:
            self.nan_indices_["categorical_has_nan_"][column] = \
                np.where(self.df_.loc[:, column].isnull())[0]
    
        for column in self.ordinal_has_nan_:
            self.nan_indices_["ordinal_has_nan_"][column] = \
                np.where(self.df_.loc[:, column].isnull())[0]

    def _fill_na(self):
        # fill in missing values
        if len(self.numeric_has_nan_) > 0:
            self.nan_numeric_ = self.numeric_has_nan_ + "_clean"

            self.df_ = self.df_.reindex(columns=[self.df_.columns.tolist() +
                                                 self.nan_numeric_.tolist()],
                                        fill_value=0)
            
            for c1, c2 in zip(self.nan_numeric_, self.numeric_has_nan_):
                self.df_.loc[:, c1] = self.df_.loc[:, c2].copy()
                self.df_.loc[:, c1] = self.df_.loc[:, c1].fillna(
                    self.df_.loc[:, c1].mean())
        else:
            self.nan_numeric_ = pd.Index(list())
        
        # no need to drop these as we're turning them into indicators anyway
        # which will drop them
        if len(self.categorical_has_nan_) > 0:
            
            self.df_.loc[:, self.categorical_has_nan_] = \
                self.df_.loc[:, self.categorical_has_nan_] \
                    .fillna(value=self.categorical_fill_value)
        else:
            self.nan_categorical_ = None
    
        # TODO: apply mapping for ordinals if present
        # TODO: we should probably look earlier for mix/max ordinal values and set the fill value based upon that to prevent collisions
        # TODO: this will have to change if we start allowing ordinal detection for non-int dtypes
        if len(self.ordinal_has_nan_) > 0:
            self.nan_ordinal_ = self.ordinal_has_nan_ + "_clean"
            self.df_ = self.df_.reindex(columns=[self.df_.columns.tolist() +
                                                 self.nan_ordinal_.tolist()],
                                        fill_value=0)

            for c1, c2 in zip(self.nan_ordinal_, self.ordinal_has_nan_):
                self.df_.loc[:, c1] = self.df_.loc[:, c2].copy()
                self.df_.loc[:, c1] = self.df_.loc[:, c1].fillna(
                    self.df_.loc[:, c1].mean())
        else:
            self.nan_ordinal_ = None
   
    def _make_indicators(self):
        # make indicators for replaced values for all dtypes
        if self.categorical_ is None:
            self.nan_categorical_ = None
        else:
            self.nan_categorical_ = self.categorical_has_nan_ + "_" + self.categorical_fill_value
            self.df_ = pd.get_dummies(self.df_,
                                      columns=self.categorical_.tolist())

    def _make_replacement_indicators(self, dataframe):
        if self.nan_numeric_ is not None:
            self.numeric_replacement_indicators_ = self.numeric_has_nan_ + \
                                                   "_replaced"

            # add the rep indicator columns to the dataframe
            self.df_ = \
                self.df_.reindex(columns=[self.df_.columns.tolist() +
                                          self.numeric_replacement_indicators_.tolist()],
                                 fill_value=0)
            
            # add the indicator if that row value was replaced
            for c1, c2 in zip(self.numeric_replacement_indicators_,
                              self.numeric_has_nan_):
                self.df_.loc[:, c1] = dataframe.loc[:, c2]\
                    .isnull()\
                    .map({True: 1,
                          False: 0})
                
            # drop the original columns with NaN
            self.df_.drop(labels=self.numeric_has_nan_.tolist(),
                          inplace=True,
                          axis=1)
            
        if self.nan_categorical_ is not None:
            self.categorical_replacement_indicators_ = self.categorical_has_nan_ + \
                                                       "_replaced"

            # add the rep indicator columns to the dataframe
            self.df_ = \
                self.df_.reindex(columns=[self.df_.columns.tolist() +
                                          self.categorical_replacement_indicators_.tolist()],
                                 fill_value=0)

            # add the indicator if that row value was replaced
            for c1, c2 in zip(self.categorical_replacement_indicators_,
                              self.categorical_has_nan_):
                self.df_.loc[:, c1] = dataframe.loc[:, c2].isnull().map(
                    {True: 1,
                     False: 0})

        if self.nan_ordinal_ is not None:
            self.ordinal_replacement_indicators_ = self.ordinal_has_nan_ + \
                                                   "_replaced"

            # add the rep indicator columns to the dataframe
            self.df_ = \
                self.df_.reindex(columns=[self.df_.columns.tolist() +
                                          self.ordinal_replacement_indicators_.tolist()],
                                 fill_value=0)

            # add the indicator if that row value was replaced
            for c1, c2 in zip(self.ordinal_replacement_indicators_,
                              self.ordinal_has_nan_):
                self.df_.loc[:, c1] = dataframe.loc[:, c2].isnull().map(
                    {True: 1,
                     False: 0})

            # drop the original columns with NaN
            self.df_.drop(labels=self.ordinal_has_nan_.tolist(),
                          inplace=True,
                          axis=1)
    
    
    