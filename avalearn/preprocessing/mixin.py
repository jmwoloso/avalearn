#!/usr/bin/env python
import numpy as np

from .base import BaseTreatmentDesign
from ..utils._validation import _check_dframe, _check_feature_target, \
    _check_train_test_size, _check_significance, _check_column_dtypes


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
                 ordinal_features=None):

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
        self.context = downstream_context
        self.deduplicate = remove_duplicates
        self.scaling = feature_scaling
        self.rare_significance = rare_level_significance
        self.feature_engineering = feature_engineering
        self.convert_dtypes = convert_dtypes
        self.ordinals = find_ordinals
        self.ordinal_features=ordinal_features

    def _validate_params(self, dataframe):
        """
        Validates the parameters passed in during intialization.
        """
        _check_dframe(dataframe=dataframe)
        self.features_, self.target_ = _check_feature_target(
            df_columns=dataframe.columns,
            features=self.features,
            target=self.target)

        self.train_size_ = _check_train_test_size(self.train_size,
                                                  self.test_size)

        self.feature_significance_, self.remove_features_ = \
            _check_significance(n_features=dataframe.shape[0],
                                significance=self.significance)

        numeric_, categorical_, ordinal_, mapping_ = \
            _check_column_dtypes(dataframe=dataframe,
                                 features=self.features_,
                                 target=self.target_,
                                 convert_dtypes=self.convert_dtypes,
                                 ordinals=self.ordinal_features,
                                 find_ordinals=self.ordinals)
    def fit(self, dataframe):
        """
        Create the treatment plan.
        """
        self._validate_params(dataframe)

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