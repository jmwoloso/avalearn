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