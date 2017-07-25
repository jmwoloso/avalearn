import pandas as pd



def _make_indicators(dataframe=None, categorical_columns=None,
                     ordinal_columns=None):
    if categorical_columns is None and ordinal_columns is None:
        return None
    elif categorical_columns is None and ordinal_columns is not None:
        return pd.get_dummies(dataframe,
                              columns=ordinal_columns.tolist())
    elif ordinal_columns is None and categorical_columns is not None:
        return pd.get_dummies(dataframe,
                              columns=categorical_columns.tolist())
    else:
        return pd.get_dummies(dataframe,
                              columns=categorical_columns.tolist() +
                                      ordinal_columns.tolist())