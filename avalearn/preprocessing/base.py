#!/usr/bin/env python
"""
base.py : Base classes for the treatment design classes.

"""
from abc import ABCMeta, abstractmethod


class BaseTreatmentDesign(object):
    """
    Base class for Treatment Design classes.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self):
