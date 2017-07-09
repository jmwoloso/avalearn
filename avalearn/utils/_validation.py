#!/usr/bin/env python
"""
_validation.py : validation routines for the avalearn package.
"""
import numpy
import pandas

# checks if we were passed a dataframe
def _check_dframe(dataframe=None):
    """
    Verifies we were passed a dataframe.
    """
    if not isinstance(dataframe, pandas.DataFrame):
        raise TypeError("`dataframe` should be of type pandas.DataFrame")
    return dataframe


def _check_feature_target(df_columns=None, features=None, target=None):
    """
    Sets the features_ and target_ attributes and verifies the specified
    features and target are in the dataframe index.
    """
    target_type = type(target)

    if target_type not in {int, str}:
        raise ValueError("`target` should be one of <int, str>".format(target_type))

    if features == "all":
        features_ = df_columns.copy()
        if target_type == str:
            if target not in features_:
                raise ValueError("`target` not found in the dataframe columns")
            else:
                features_.drop(target)
                target_ = target
        else:
            target_ = features_[target]
            features_ = features[:target] if target > 0 else features[target + 1:]
    else:
        features_type = type(features)

        if features_type != list:
            raise ValueError("`features` should be one of <list of int, list of str, 'all'>")
        else:
            f0_type = type(features[0])
            if f0_type not in {int, str}:
                raise ValueError("`features` contains values which cannot be "
                                 "used for label indexing")
            f1_type = type(features[1])
            if f0_type != f1_type:
                raise ValueError("`features` contains a mix of label types "
                                 "'{0}' and '{1}' and cannot be used "
                                 "for indexing".format(f0_type, f1_type))

            if f0_type == str:
                features_ = df_columns.copy()
                if target_type == str:
                    if target not in features_:
                        raise ValueError("`target` not found in the dataframe columns")
                    else:
                        features_.drop(target)
                        target_ = target
                else:
                    target_ = features_[target]
                    features_ = features[:target] if target > 0 else features[target + 1:]

            if f0_type == int:
                features_ = df_columns.copy()
                if target_type == str:
                    if target not in df_columns:
                        raise ValueError("`target` not found in the dataframe columns")
                    else:
                        features_.drop(target)
                        target_ = target
                else:
                    target_ = features_[target]
                    features_ = features[:target] if target > 0 else features[target + 1:]
    return features_, target_