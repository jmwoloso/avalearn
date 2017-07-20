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
        for i, value in enumerate(values):
            mapping[value] = i
        return mapping

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
                    dataframe.loc[:, column].dropna().astype(int)
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


def _check_positive_class(class_value=None, target_classes=None):
    if class_value not in target_classes:
        raise ValueError("`positive_class` not found in `target`")


def _check_keyword_type(keyword=None, parameter=None, values=None,
                         keyword_type=None, min_max_range=None):
    if not isinstance(keyword, keyword_type):
        raise ValueError("`{0}` must be one of [{1}]"
                         .format(parameter,
                                 ", ".join(str(item) for item in values)))
    if min_max_range is not None:
        if not min_max_range[0] < keyword < min_max_range[1]:
            raise ValueError("`{0}` must be one of [{1}]"
                             .format(parameter,
                                     ", ".join(str(item) for item in values)))
    return True


def _check_keywords(keyword=None, parameter=None, values=None,
                    keyword_type=None, min_max_range=None):
    if keyword_type is not None:
        _check_keyword_type(keyword, parameter, values, keyword_type,
                            min_max_range)

    elif keyword not in values:
        raise ValueError("`{0}` must be one of [{1}]"
                         .format(parameter, ", ".join(str(item) for item in
                                                      values)))


def _check_boolean(boolean=None, parameter=None):
    if not isinstance(boolean, bool):
        raise ValueError("`{}` must be one of [True, False]".format(parameter))