#!/usr/bin/env python
import numpy as np

from .base import BaseTreatmentDesign



class TreatmentDesignMixin(BaseTreatmentDesign):
    """
    Mixin class defining core functionality for treatment design-related
    classes.
    """

    def __init__(self, features="all", target=-1,
                 min_feature_significance=None, cv=3, cv_type='loo',
                 cv_split_function=None, train_size=0.2,
                 test_size=None, random_state=None,
                 rare_level_min_fraction=0.02,
                 novel_level_strategy="nan", rare_level_pooling=False,
                 make_nan_indicators=True,
                 high_cardinality_strategy="impact",
                 downstream_context=None, remove_duplicates=False,
                 feature_scaling=False,
                 rare_level_significance=None,
                 feature_engineering=None,
                 convert_dtypes=True):

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
        self.rare_fraction = rare_level_min_fraction
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


    def fit(self, dataframe):
        """
        Create the treatment plan.
        """
        pass

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