# -*- coding: utf-8 -*-
"""
Gradient Descent
"""
from cost import calculate_mse


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err
