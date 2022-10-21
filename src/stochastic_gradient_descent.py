# -*- coding: utf-8 -*-
"""
Stochastic Gradient Descent

"""
from helpers import batch_iter


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient for batch data."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err


