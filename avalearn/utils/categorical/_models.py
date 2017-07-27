"""
_models.py : modeling routines for determining variable significance
"""
import numpy as np

def logit(x):
    return np.log(x/(1-x))


def _get_conditional_scores(dataframe=None, feature=None, target=None,
                            positive_class=None, print_output=True,
                            master_dict=None):
    """
    Creates a single-variable Bayesian model from the provided feature. The
    feature is an indicator created from a particular level of the original
    categorical feature.
    """
    # fillna so we can use all category levels
    # dataframe.loc[:, feature
    dataframe.loc[:, feature] = dataframe.loc[:, feature].fillna(value="NaN")
    unique_levels = dataframe.loc[:, feature].unique()
    unique_count = len(unique_levels) - 1
    # extra_degrees_of_freedom = np.max([0, unique_count])
    # epsilon = 1.0e-6
    smoothing_factor = 1.0e-4

    # conditions used to derive modeling values
    # level_and_true = dataframe.loc[:, feature] == positive_class
    overall_true = dataframe.loc[:, target] == positive_class
    overall_false = dataframe.loc[:, target] != positive_class
    
    # number of True samples
    n_true = dataframe[overall_true].loc[:, target].count()
    n_false = dataframe[overall_false].loc[:, target].count()
    
    # unconditioned mean
    probability_of_true = dataframe.loc[:, target].mean()
    
    # container for conditional scores for each level
    conditional_scores = dict()
    master_dict[feature] = dict()
    # find the conditional scores for each unique category level
    for level in dataframe.loc[:, feature].unique():
        # Original version: caused boolean index warning
        # condition = dataframe.loc[:, feature] == level
        # count of True for the current category level
        # n_conditional_and_true = dataframe[condition][overall_true].loc[:, feature].count()
        # count of False for the current category level
        # n_conditional_and_false = dataframe[condition][overall_false].loc[:, feature].count()
        
        true_filter = "{0} == '{1}' & {2} == {3}".format(feature,
                                                         level,
                                                         target,
                                                         positive_class)

        false_filter = "{0} == '{1}' & {2} != {3}".format(feature,
                                                          level,
                                                          target,
                                                          positive_class)

        n_conditional_and_true = dataframe.query(true_filter).loc[:, feature].count()
        n_conditional_and_false = dataframe.query(false_filter).loc[:, feature].count()
        
        # probability of category level given the outcome is the positive_class
        prob_of_level_given_true = (n_conditional_and_true + smoothing_factor)/\
                                   (n_true + smoothing_factor)
        # probability of category level given the outcome is NOT the positive_class
        prob_of_level_given_false = (n_conditional_and_false + smoothing_factor) /\
                                    (n_false + smoothing_factor)

        # un-normalized probability the target is the positive class given the level
        prob_true_given_level_unnormalized = prob_of_level_given_true * probability_of_true
        # un-normalized probability the target is NOT the positive class given the level
        prob_false_given_level_unnormalized = prob_of_level_given_false * (1 -
                                                                           probability_of_true)
    
        # normalized probability the target is the positive class given the level
        prob_of_true_given_level = prob_true_given_level_unnormalized / \
                                   (prob_true_given_level_unnormalized + prob_false_given_level_unnormalized)

        # find the conditional score for this level
        conditional_score = logit(prob_of_true_given_level) - logit(probability_of_true)
        
        # add the score to the container
        conditional_scores[level] = conditional_score
        
        if print_output is True:
            print("n_true: {0}".format(n_true))
            print("n_false: {0}".format(n_false))
            print("conditional_n_true: {0}".format(n_conditional_and_true))
            print("conditional_n_false: {0}".format(n_conditional_and_false))
            print("probability_of_true: {0}".format(probability_of_true))
            print("prob_of_level_given_true: {0}".format(prob_of_level_given_true))
            print("prob_of_level_given_false: {0}".format(prob_of_level_given_false))
            print("prob_of_true_given_level_unnormalized: {0}".format(prob_true_given_level_unnormalized))
            print("prob_of_false_given_level_unnormalized: {0}".format(prob_false_given_level_unnormalized))
            print("prob_of_true_given_level: {0}".format(prob_of_true_given_level))
            print("conditional_score: {0}".format(conditional_score))

    master_dict[feature] = conditional_scores
    return master_dict


