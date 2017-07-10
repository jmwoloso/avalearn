#!/usr/bin/env python
"""
_validation.py : validation routines for the avalearn package.
"""
import pandas


def _check_dframe(dataframe=None):
    """
    Verifies we were passed a dataframe.
    """
    if not isinstance(dataframe, pandas.DataFrame):
        raise TypeError("`dataframe` should be of type pandas.DataFrame")


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


def _check_ordinal_features(features=None):
    """
    Attempts to find ordinal features among the features and converts the
    dtype to 'category'.
    """
    pass


def _check_column_dtypes(dataframe=None, features=None, target=None,
                         convert_dtypes=True,
                         ordinals=None, find_ordinals=False):
    """
    Separates numeric from categorical columns; attempts to convert
    categorical columns to numeric if `find_ordinals=True`.
    """
    def mapper(values):
        mapping = dict()
        for i, value in enumerate(range(len(values))):
            mapping[value] = i
        return mapping
    # if convert_dtypes not in {True, False}:
    #     raise ValueError("`convert_dtypes` must be one of [True, False]")
    if find_ordinals not in {True, False}:
        raise ValueError("`find_ordinals` must be one of [True, False]")
    if find_ordinals is True:
        if ordinals != None:
            find_ordinals = False
        else:
            # find int dtypes
            ordinal_ = dataframe.loc[:, dataframe.dtypes ==
                                        int].columns.tolist()

            for column, _ in dataframe.loc[:, dataframe.dtypes ==
                    object].iteritems():
                try:
                    dataframe.loc[:, column].astype(int)
                    ordinal_.append(column)
                except ValueError:
                    pass
    numeric_ = dataframe.loc[:, features].select_dtypes(include=['float']).columns
    categorical_ = dataframe.loc[:, ~dataframe.columns.isin(ordinal_)].select_dtypes(include=['object']).columns
    # create the mapping
    mapping_ = dict()
    for column, _ in dataframe.loc[:, categorical_].iteritems():
        mapping_[column] = dict()
        mapping_[column] = mapper(dataframe.loc[:, column].unique().tolist())
    return numeric_, categorical_, pandas.Index(ordinal_), mapping_


def _check_train_test_size(train_size=None, test_size=None):
    """
    Validates the values passed for train_size or test_size and sets the
    train_size_ and test_size_ attributes.
    """
    if test_size is not None:
        if not isinstance(test_size, float):
            raise TypeError("`test_size` should be of type <float>")
        if not 0.0 < test_size < 1.0:
            raise ValueError("`test_size` should be a float in the interval "
                             "(0.0, 1.0)")
        test_size_ = test_size
        train_size_ = 1 - test_size_

    if train_size is not None:
        if not isinstance(train_size, float):
            raise TypeError("`train_size` should be of type: float")
        if not 0.0 < train_size < 1.0:
            raise ValueError("`train_size` should be in the interval (0.0, "
                             "1.0)")
        train_size_ = train_size
        test_size_ = 1 - train_size_

    return train_size_, test_size_


def _check_significance(n_features=None, significance=None):
    """
    Validates the min_feature_significance parameter and sets the
    feature_significance and remove_features_ attributes.
    """
    if significance is None:
        feature_significance_ = significance
    if significance == "1/n_features":
        feature_significance_ = 1.0 / n_features
    sig_type = type(significance)
    if sig_type == str:
        raise ValueError("`min_feature_significance` should be one of [float in the interval (0, 1), '1/n_features', None]")
    if sig_type != float:
        raise ValueError("`min_feature_significance` should be one of [float in the interval (0, 1), '1/n_features', None]")
    if sig_type == float and not 0.0 < significance < 1.0:
        raise ValueError("`min_feature_significance` should be type <float> "
                         "in the interval (0.0, 1.0)")

    remove_features_ = False if feature_significance_ is None else True

    return feature_significance_, remove_features_


def _validate_init_params(dataframe=None, features=None, target=None,
                          convert_dtypes=None, train_size=None,
                          test_size=None, significance=None, ordinals=None,
                          find_ordinals=False):
    """
    Master validation function that dispatches the input parameters for
    checking then records the appropriate class attributes.
    """
    _check_dframe(dataframe=dataframe)
    features_, target_ = _check_feature_target(df_columns=dataframe.columns,
                                               features=features,
                                               target=target)

    train_size_ = _check_train_test_size(train_size, test_size)

    feature_significance_, remove_features_ = \
        _check_significance(n_features=dataframe.shape[0],
                            significance=significance)

    numeric_, categorical_, ordinal_, mapping_ = \
        _check_column_dtypes(dataframe=dataframe,
                             features=features_,
                             target=target_,
                             convert_dtypes=convert_dtypes,
                             ordinals=ordinals,
                             find_ordinals=find_ordinals)