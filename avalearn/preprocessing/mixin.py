#!/usr/bin/env python
import numpy as np
from sklearn.utils import check_random_state
from .base import BaseTreatmentDesign, TreatmentDescriptionDataFrame, \
    DelayedDataFrame
from ..utils._validation import _check_dframe, _check_feature_target, \
    _check_train_test_size, _check_column_dtypes, \
    _check_boolean, _check_keywords


class TreatmentDesignMixin(BaseTreatmentDesign):
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
                 feature_engineering=None,
                 convert_dtypes=True,
                 find_ordinals=True,
                 ordinals=None,
                 high_cardinality_threshold=None,
                 n_jobs=1,
                 ordinal_mapping=None):

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

    def _validate_params(self, dataframe):
        """
        Validates the parameters passed in during initialization.
        """
        _check_dframe(dataframe=dataframe)

        _check_keywords(self.significance,
                        "min_feature_significance",
                        ["float in range(0, 1)", "1/n_features", None],
                        float,
                        [0,1])

        _check_keywords(self.unique_percent,
                        "unique_level_min_percent",
                        ["float in range(0, 1)"],
                        float,
                        [0,1])

        # TODO: add enhancement to allow value to be set based upon the dataset instead of fixed int values
        _check_keywords(self.high_cardinality_threshold,
                        "high_cardinality_threshold",
                        [int],
                        int,
                        [0, np.inf])

        _check_keywords(self.n_jobs,
                        "n_jobs",
                        [int],
                        int,
                        [-2, np.inf])

        _check_keywords(self.context, "downstream_context", ["pipeline", None])

        _check_keywords(self.high_cardinality_strategy,
                        "high_cardinality_strategy",
                        ["impact", "indicators"])

        _check_keywords(self.mapping,
                        "ordinal_mapping",
                        [dict, None],
                        dict,
                        None)

        _check_keywords(self.novel_strategy,
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

        self.numeric_, self.categorical_, self.ordinal_, self.mapping_ = \
            _check_column_dtypes(dataframe=dataframe,
                                 features=self.features_,
                                 target=self.target_,
                                 convert_dtypes=self.convert_dtypes,
                                 ordinals=self.ordinals,
                                 find_ordinals=self.find_ordinals)
        
        self.random_state_ = check_random_state(self.random_state)








        return self


    def fit(self, dataframe):
        """
        Create the treatment plan.
        """
        # input validation
        self._validate_params(dataframe)
        # with input validated, create the TreatmentDescriptionDF object
        self.TreatmentDescriptionDF = TreatmentDescriptionDataFrame()
        self.DelayedDF = DelayedDataFrame


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