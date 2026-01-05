import numpy as np
from scipy.stats import spearmanr, pearsonr, kendalltau

def calculate_srcc(pred, target):
    """
    Calculate Spearman Rank Correlation Coefficient.
    """
    pred = np.array(pred).flatten()
    target = np.array(target).flatten()
    return spearmanr(pred, target)[0]

def calculate_plcc(pred, target):
    """
    Calculate Pearson Linear Correlation Coefficient.
    """
    pred = np.array(pred).flatten()
    target = np.array(target).flatten()
    return pearsonr(pred, target)[0]

def calculate_krcc(pred, target):
    """
    Calculate Kendall Rank Correlation Coefficient.
    """
    pred = np.array(pred).flatten()
    target = np.array(target).flatten()
    return kendalltau(pred, target)[0]
