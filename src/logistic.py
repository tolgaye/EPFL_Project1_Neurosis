import numpy as np
import math
from typing import  Tuple
from random import randrange


def sigmoid(t):
    """
    Applies the sigmoid function to a given input t.
    Args:
        t: the given input to which the sigmoid function will be applied.
    Returns:
        sigmoid_t: the value of sigmoid function applied to t
    """
    return 1. / (1. + np.exp(-t))


def logistic_2(y, tx, w, lambda_):
    """
    Compute logistic loss with thresholds to avoid computational overflow
    :param y: labels
    :param tx: features
    :param w: weights
    :param lambda_: Regularizer factor
    """
    in_pred = tx.dot(w)
    in_pred[in_pred >= 10] = 10
    in_pred[in_pred <= -10] = -10
    pred = sigmoid(in_pred)
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(- loss) + lambda_ * np.squeeze(w.T.dot(w))


def compute_logistic_loss(y, tx, w, loss_function='mse', lambda_=0):
    """Calculate the loss given a specific function and an optional regularizer factor.
    :param y: labels
    :param tx: features
    :param w: weights
    :param loss_function: Loss function, possibilities specified below
    :param lambda_: Regularizer factor
    The possible loss functions are:
        * MSE (By default)
        * MAE
        * RMSE (Root MSE)
        * Logistic
    """
    return {
        'mse': 1 / (2 * len(y)) * sum((y - tx.dot(w)) ** 2) + lambda_ * np.linalg.norm(w) ** 2,
        'rmse': 1 / (len(y)) * np.abs(sum((y - tx.dot(w)))) + lambda_ * np.linalg.norm(w) ** 2,
        'mae': 1 / len(y) * sum(np.abs(y - tx.dot(w))) + lambda_ * np.linalg.norm(w) ** 2,
        'logistic': logistic_2(y, tx, w, lambda_)
    }[loss_function]


def compute_logistic_gradient(y, tx, w, loss_function='mse', lambda_=0):
    """Compute a stochastic gradient from just few examples n and
    their corresponding y_n labels.
    :param y: labels
    :param tx: features
    :param w: weights
    :param loss_function: Loss function, possibilities specified below
    :param lambda_: Regularizer factor
    The possible loss functions are:
        * MSE (By default)
        * MAE
        * RMSE (Root MSE)
        * Logistic
    """
    return {
        'mse': -1 / len(y) * tx.transpose().dot(y - tx.dot(w)) + 2 * lambda_ * w,
        'rmse': -1 / np.sqrt(len(y)) * tx.transpose().dot([-1 if e <= 0 else 1 for e in (y - tx.dot(w))])
                + 2 * lambda_ * w,
        'mae': -1 / len(y) * tx.transpose().dot([-1 if e <= 0 else 1 for e in (y - tx.dot(w))]) + 2 * lambda_ * w,
        'logistic': tx.T.dot(sigmoid(tx.dot(w)) - y) + 2 * lambda_ * w
    }[loss_function]
    
    def gradient_descent_step(y: np.ndarray, tx: np.ndarray, w: np.ndarray, gamma: float,
                          lambda_: float = 0, mode='linear') -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Computes one step of gradient descent.
    Parameters
    ----------
    y: ndarray
        Array that contains the correct values to be predicted.
    tx: ndarray
        Matrix that contains the data points. The first column is made of 1s.
    w: ndarray
        Array containing the linear parameters to test.
    gamma: float
        The stepsize.
    lambda_: float
        The lambda used for regularization. Default behavior is without regularization.
    
    mode: str
        The kind of gradient descent. Must be either `linear` or `logistic`
    Returns
    -------
    w: np.ndarray
        The linear parameters.
    loss: float
        The loss given w as parameters.
    """
    # Get loss, gradient, hessian
    loss = (compute_loss(y, tx, w, lambda_=lambda_) if mode == 'linear'
            else compute_logistic_loss(y, tx, w, lambda_=lambda_))
    
    gradient = (compute_gradient(y, tx, w) if mode == 'linear'
               else compute_logistic_gradient(y, tx, w, lambda_=lambda_))

    # Update w
    w = w - gamma * gradient

    return loss, gradient, w
