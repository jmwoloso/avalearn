#!/usr/bin/env python
"""
data_prep.py: classes for designing data treatment.
"""
import pandas as pd

from .mixin import TreatmentDesignMixin
from ..utils._validation import _check_positive_class


class ClassificationTreatmentDesign(TreatmentDesignMixin):
    """
    Class for designing treatments for classification tasks.
    """
    def __init__(self, features="all", target=-1, ordinals=None,
                 min_feature_significance=None, find_ordinals=True,
                 ordinal_mapping=None,
                 unique_level_min_percent=0.02, positive_class=1,
                 cv=3, cv_type='loo', cv_split_function=None, train_size=0.2,
                 test_size=None, random_state=None, n_jobs=1,
                 novel_level_strategy="nan", make_nan_indicators=True,
                 rare_level_pooling=False,
                 high_cardinality_strategy="impact",
                 downstream_context=None, remove_duplicates=False,
                 feature_scaling=False, rare_level_significance=None,
                 feature_engineering=None, high_cardinality_threshold=None):
        """

        Parameters
        ----------
        features : one of {list of column names, list of column indices, "all"}; default="all"
            The columns that should be used during treatment design.

        target : one of {target column name; target column index}; default=-1
            The column name or column index where the target resides.

        ordinals : one of {list of column names, list of column indices, "all", None}; default=None
            The column names or column indices for any known ordinal
            features. Features of this type can be safely hashed to numeric
            values for use in downstream machine learning models. Ordinal
            features are still subject to the parameter value set for
            `rare_level_threshold` and `rare_level_pooling` and will be
            handled accordingly after they are hashed.

        min_feature_significance : one of {float in the interval (0, 1), "1/n_features", None}; default="1/n_features"
            If `None`, no feature pruning will take place otherwise this is
            the minimum significance level a feature needs in order to be
            included in the final treated dataframe where lower values
            indicate more significance.

        unique_level_min_percent : float in range(0,1); default=0.02
            The minimum percentage of time that a specific level of a
            categorical variable has to show up in order to be treated as a
            unique level. Any categorical level failing to meet the
            threshold will be pooled into a new categorical level
            representing all rare levels.

        rare_level_pooling : bool; default=False
            Whether to pool rare categorical levels together into a single
            pooled level; "rare" levels are defined using the value specified
            for `rare_level_threshold`.

        high_cardinality_strategy : str, one of {"impact", "indicators"}; default="impact".
            Whether to create indicator variables for categorical levels
            with high cardinality (many levels) or use impact coding; this
            strategy takes effect once the number of levels for a
            categorical feature meets or exceeds `high_cardinality_threshold`.

        high_cardinality_threshold : int; default=20
            The maximum levels that a categorical variable may contain
            before it is flagged as being high cardinality.

        positive_class : int; default=1
            The value of the positive class for regression tasks.

        cv : int or str; one of {int, 'n'}; default=3
            If an int is supplied this is the number of cross-validation
            folds to use when designing treatment.

            If 'n' is supplied, leave-one-out cross-validation will be
            employed where each sample is used exactly once as the test set
            while the remaining samples will be used as the training set for
            model building and treatment design.

        cv_type : str; one of {'kfold', 'stratified', 'loo'}
            Specifies the cv strategy to use for designing treatment.

        cv_split_function : default=None
            Allows specifying a custom splitting function for creating the cv
            splits.

        train_size : float in interval (0,1); default=0.2.
            Specifies the size of the training sets when using cross-validation
            to design treatment; ignored if `test_size` is specified.

        test_size : float in the interval (0, 1); default=None.
            Specifies the size of the test sets when using cross-validation
            to design treatments; ignored if `train_size` is specified.

        random_state : one of {int, None}; default=None
            Allows splits and results of treatment design to be re-producible.

        n_jobs : int or -1; default=1
            Number of cpu cores to use when designing and applying treatment.

        make_nan_indicators : bool; default=True

        novel_level_strategy : one of {"known", "zero", "nan", "rare", "pooled"}; default="nan"
            The strategy used when novel categorical levels are encountered
            during data treatment.

              'known' : novel levels are weighted proportional to known levels.

              'zero' : novel levels are treated as "no level" and assigned
              zero for known-level indicators.

              'nan' : novel levels are treated as missing and assigned to the
              "NaN" level indicators if `make_nan_indicators`=True.

              'rare' : novel levels are weighted proportional to rare levels
              only.

              'pooled' : novel levels are added to the pooled indicator level
              IF `rare_level_pooling`=True.

        downstream_context : one of {"pipeline", None}; default=None
            The context in which the data will be used downstream. Setting
            this parameter to "pipeline" will result in transformed data
            being returned in a format suitable for use in the
            sklearn.pipeline.Pipeline class.

        positive_class : one of {int, float, str}; default=None
            The class label representing the positive case.

        find_ordinals : bool; default=False
            Whether to attempt to find ordinals among the features. Ignored
            when `ordinal_features` != None. Ordinals are found naively by
            checking first for columns where dtype == int and then by
            attempting to coerce columns where dtype == object into dtype == int

            Note: this is distinct behavior relative to the
            `ordinal_mapping` parameter as this method applies to finding
            ordinal features among the dtype == int columns while the ordinal
            mapping relates to columns where dtype == object but are known to
            contain an ordering among the levels that must be supplied in order
            to consider the column(s) to be ordinal.

        ordinal_mapping : one of {nested dict of mappings, None}
            If `ordinal_features` != None and a dict is provided, the dict
            should contain the ordinal column names as top-level keys and
            within each top-level key, another dict should reside containing the
            (sorted) original values as keys and their appropriate mapping
            (to ints). The mapping will be applied to the specified columns
            and preserved in the `ordinal_mapping_` attribute.

            The mapping dict should be like the following:

            map_dict = {"ColumnA": {"A": 0,
                                    "B": 1}}

            If `ordinal_features` = None and `find_ordinals` = True, each feature
            column where dtype=object will be checked and mapped accordingly
            with the mapping preserved in the `ordinal_mapping_` attribute.

            If `find_ordinals`=True, the algorithm looks first for
            columns with dtype=int and will re-classify them as
            dtype=category without creating a mapping as the mapping is
            thought to be inherent in the values.

            For columns of dtype=object, an attempt to convert them to
            dtype=int will be made and if successful, no mapping will be
            created for the same reasons as above. If they cannot be
            converted to dtype=int, they will be sorted alphabetically and
            mapped accordingly.

            Note: this feature may create undesirable mappings in some
            cases, such as columns where the logical order is something
            other than alphabetic, e.g. {'XS', 'S', 'M', 'L', 'XL'}.

        remove_duplicates : bool; default=True
            Whether to remove duplicate columns or not. During treatment
            design duplicate columns will be flagged but will not be removed
            until the treatment is applied to the data via the `fit` method.

        feature_engineering : bool; default=False
            Whether to create interaction terms and apply additional feature
            engineering steps to the dataframe.

        feature_scaling : bool; default=True
            Whether to standardize the data.

        Attributes
        ----------
        df_ : pandas.DataFrame.
            The original pandas.DataFrame used for treatment design.

        features_ : pandas.DataFrame.
            The subset of features that need treating.

        target_ : pandas.DataFrame.
            The target column.

        numeric_ : pandas.DataFrame.
            The numeric features in the dataframe.

        categorical_ : pandas.DataFrame.
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
        # TODO: Raise NotImplementedError for feature engineering (initially)
        # TODO: check for multiclass classification (count of unique class labels and raise NotImplementedError for the time being
        # TODO: update attributes
        super().__init__(features, target, ordinals, n_jobs,
                         min_feature_significance, find_ordinals,
                         cv, cv_type, cv_split_function, train_size, test_size,
                         random_state, novel_level_strategy,
                         unique_level_min_percent, rare_level_pooling,
                         rare_level_significance, feature_engineering,
                         make_nan_indicators, high_cardinality_strategy,
                         downstream_context, remove_duplicates,
                         feature_scaling, high_cardinality_threshold, ordinal_mapping)
        self.positive_class = positive_class


    def fit(self, dataframe):
        """
        Analyze the dataframe, gather descriptive information and design the treatment.

        Parameters
        ----------
        dataframe : pandas.DataFrame instance.
            The dataframe to be used for treatment design.

        Returns
        -------
        self : object.
            Returns self.
        """
        super().fit(dataframe)
        _check_positive_class(self.positive_class, self.target_)


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


class RegressionTreatmentDesign(TreatmentDesignMixin):
    """
    Class for designing treatments for regression tasks.
    """
    def __init__(self, feature_columns="all", target_column=-1,
                 min_feature_significance=0.05, cv=3,
                 cv_split_function=None, random_state=None):
        """

        Parameters
        ----------
        treatment_columns : one of {list of column names, list of column indices, "all"}; default="all"
            The columns that should be used during treatment design.

        target_column : one of {target column name; target column index}; default=-1
            The column name or column index where the target resides.

        min_feature_significance : one of {float in range(0,1), None}; default=0.05
            If `None`, no feature pruning will take place otherwise this is
            the minimum significance level a feature needs in order to be
            included in the final treated dataframe where lower values
            indicate more significance.

        cv : int; default=3
            Number of cross-validation folds to use when designing treatment.

        cv_split_function : default=None
            Allows specifying a custom splitting function for creating the cv
            splits.

        random_state : one of {int, None}; default=None
            Allows splits and results of treatment design to be re-producible.

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
        super().__init__(feature_columns, target_column,
                         min_feature_significance, cv, cv_split_function)

    def fit(self, dataframe):
        """
        Analyze the dataframe, gather descriptive information and design the treatment.

        Parameters
        ----------
        dataframe : pandas.DataFrame instance.
            The dataframe to be used for treatment design.

        Returns
        -------
        self : object.
            Returns self.
        """
        super().fit(dataframe)


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