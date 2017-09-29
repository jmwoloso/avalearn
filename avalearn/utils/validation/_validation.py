#!/usr/bin/env python
"""
_validation.py : validation routines for the avalearn package.
"""
import pandas

# TODO: make sure this stays up-to-date
NOT_IMPLEMENTED = ["feature_engineering",
                   "cv_split_function",
                   "rare_level_significance",
                   "convert_dtypes"]

# general purpose validation functions
def _check_dframe(dataframe=None):
    """
    Verifies we were passed a dataframe.
    """
    if not isinstance(dataframe, pandas.DataFrame):
        raise TypeError("`dataframe` should be of type pandas.DataFrame")


def _check_column_dtypes(dataframe=None, features=None, target=None,
                         convert_dtypes=True, ordinals=None,
                         find_ordinals=False, predefined_mapping=None,
                         categorical_fill_value=None):
    """
    Separates numeric from categorical columns; attempts to convert
    categorical columns to numeric if `find_ordinals=True`.
    """
    # TODO: make use of or remove features, convert_dtypes and predef mapping
    def mapper(values, fill_value):
        mapping = dict()
        for i, value in enumerate(values):
            if pandas.isnull(value):
                mapping[fill_value] = i
            else:
                mapping[value] = i
        return mapping
    
    columns = dataframe.columns
    
    try:
        columns = columns.drop(labels=target)
    except ValueError:
        pass
    
    if find_ordinals is True:
        if ordinals != None:
            find_ordinals = False
        else:
            # find int dtypes
            ordinal_ = dataframe.loc[:, columns].loc[:,
                       dataframe.dtypes == int].columns.tolist()
            
            for column, _ in dataframe.loc[:, columns].loc[:,
                             dataframe.dtypes == object].iteritems():
                try:
                    dataframe.loc[:, column].dropna().astype(int)
                    ordinal_.append(column)
                except ValueError:
                    pass
            
            # also check whether floats are whole numbers that can be safely converted to int
            # TODO: ENH: we could be more aggressive in declaring float columns as ordinal
            for column, _ in dataframe.loc[:, columns].loc[:,
                             dataframe.dtypes == float].iteritems():
                is_int_sum = dataframe.loc[:, column].apply(
                    lambda x: x.is_integer()).sum()
                # if EVERY non-nan value doesn't satisfy is_integer(),
                # we'll leave the column as float to be safe
                if is_int_sum == dataframe.loc[:, column].dropna().shape[0]:
                    ordinal_.append(column)
                else:
                    pass
    
    numeric_ = dataframe.loc[:, ~dataframe.columns.isin(ordinal_)] \
        .select_dtypes(include=['float']) \
        .columns
    
    categorical_ = dataframe.loc[:, ~dataframe.columns.isin(ordinal_)] \
        .select_dtypes(include=['object']) \
        .columns
    
    # create the mapping
    mapping_ = dict()
    for column, _ in dataframe.loc[:, categorical_].iteritems():
        mapping_[column] = dict()
        mapping_[column] = mapper(dataframe.loc[:, column].unique().tolist(),
                                  categorical_fill_value)
    
    # find columns with nan values
    nan_numeric_ = dataframe.loc[:, numeric_].loc[:,
                   dataframe.isnull().any()].columns
    nan_categorical_ = dataframe.loc[:, categorical_].loc[:,
                       dataframe.isnull().any()].columns
    nan_ordinal_ = dataframe.loc[:, ordinal_].loc[:,
                   dataframe.isnull().any()].columns
    
    return numeric_, categorical_, pandas.Index(ordinal_), mapping_, \
           nan_numeric_, nan_categorical_, nan_ordinal_


def _check_boolean(boolean=None, parameter=None):
    if not isinstance(boolean, bool):
        raise ValueError("`{}` must be one of [True, False]".format(parameter))


def _check_int_keyword(keyword=None, parameter=None, numeric_range=None):
    if type(keyword) == int:
        if numeric_range is None:
            return True
        elif not numeric_range[0] <= keyword < numeric_range[1]:
            raise ValueError("`{0}` type must be <int> in range({1},{2})"
                             .format(parameter,
                                     numeric_range[0],
                                     numeric_range[1]))
    else:
        raise ValueError("`{0}` type must be <int>"
                         .format(parameter))


def _check_str_keyword(keyword=None, parameter=None, str_values=None):
    if keyword is None:
        return
    if type(keyword) != str:
        raise ValueError("`{0}` must be one of [{1}]"
                         .format(parameter,
                                 ", ".join(
                                     str(item) for item in str_values)))
    if keyword not in str_values:
        raise ValueError("`{0}` must be one of [{1}]"
                         .format(parameter,
                                 ", ".join(str(item) for item in str_values)))
    
    
def _check_float_keyword(keyword=None, parameter=None, keyword_range=None):
    if type(keyword) != float:
        raise ValueError("`{0}` type must be <float> in range({1},{2})"
                         .format(parameter,
                                 keyword_range[0],
                                 keyword_range[1]))
    
    if keyword_range is None:
            return
    else:
        if not keyword_range[0] < keyword < keyword_range[1]:
            raise ValueError("`{0}` type must be <float> in range({1},{2})"
                             .format(parameter,
                                     keyword_range[0],
                                     keyword_range[1]))

# keyword-specific validation functions
def _check_min_feature_significance(keyword, param_values):
    parameter="min_feature_significance"
    if keyword is None:
        return
    elif type(keyword) not in [float, str]:
        raise ValueError("`{0}` must be one of [{1}]"
                         .format(parameter,
                                 ", ".join(str(val) for val in param_values)))
    if type(keyword) == float:
        if not 0.0 < keyword < 1.0:
            raise ValueError("`{0}` must be one of [{1}]"
                             .format(parameter,
                                     ", ".join(str(val) for val in param_values)))
    if type(keyword) == str:
        if keyword != "1/n_features":
            raise ValueError("`{0}` must be one of [{1}]"
                             .format(parameter,
                                     ", ".join(
                                         str(val) for val in param_values)))


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


def _check_hc_threshold(keyword=None, param_values=None):
    parameter="high_cardinality_threshold"
    if keyword is None:
        return
    elif type(keyword) != int:
        raise ValueError("`{0}` type must be one of [int >= 0, None]"
                         .format(parameter,
                                 ", ".join(str(val) for val in param_values)))


def _check_ordinal_mapping(keyword=None, param_types=None, df_columns=None):
    parameter="ordinal_mapping"
    if keyword is None:
        return
    if type(keyword) != dict:
        raise ValueError("`{0}` type must be one of [{1}]"
                         .format(parameter,
                                 ", ".join(str(val) for val in param_types)))
    for key in keyword.keys():
        if key not in df_columns:
            raise ValueError("column `{0}` not found in the dataframe "
                             "columns".format(key))


def _check_cat_fill_value(keyword="NaN", parameter=None):
    if keyword is None:
        raise ValueError("`{0}` type must be <str>"
                         .format(parameter))
    if type(keyword) != str:
        raise ValueError("`{0}` type must be <str>"
                         .format(parameter))
    
    
    
    def _check_str_keyword(keyword=None, parameter=None, str_values=None):
        if keyword is None:
            return
        if type(keyword) != str:
            raise ValueError("`{0}` must be one of [{1}]"
                             .format(parameter,
                                     ", ".join(
                                         str(item) for item in str_values)))
        if keyword not in str_values:
            raise ValueError("`{0}` must be one of [{1}]"
                             .format(parameter,
                                     ", ".join(
                                         str(item) for item in str_values)))



def _check_mixed_type_keyword(keyword="", value="", types=(),
                              types_dict={}):
    if not isinstance(value, types):
        raise ValueError("`{}` must be one of {}".format(keyword,
                                                         value))
    elif isinstance(value, list):
        pass
    elif isinstance(value, dict):
        pass
    elif isinstance(value, str):
        pass
    elif isinstance(value, int):
        pass
    elif isinstance(value, float):
        pass
    elif isinstance(value, bool):
        pass
    elif keyword in NOT_IMPLEMENTED:
        raise NotImplementedError("`{}` is not currently "
                                  "implemented".format(keyword))




