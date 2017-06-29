#!/usr/bin/env python
"""
data_prep.py: classes for designing data treatment.
"""
import pandas as pd

from .base import TreatmentDesignMixin



class ClassificationTreatmentDesign(TreatmentDesignMixin):
    """
    Class for designing treatments for classification tasks.
    """
    def __init__(self, feature_columns="all", target_column=-1,
                 min_feature_significance=None, positive_class=1):
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

        positive_class : int; default=1; required.
            The value of the positive class for regression tasks.

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
        # TODO: check for multiclass classification (count of unique class labels and raise NotImplementedError for the time being
        super().__init__(feature_columns, target_column, min_feature_significance)
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
        super().__init__(feature_columns, target_column, min_feature_significance)

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