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

def compute_gradient(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Computes the gradient of the MSE
    Parameters
    ----------
    y: ndarray
        Array that contains the correct values to be predicted.
    tx: ndarray
        Matrix that contains the data points. The first column is made of 1s.
    w: ndarray
        Array containing the parameters of the linear model, from w0 on.
    Returns
    -------
    gradient: ndarray
        Array containing the gradient of the MSE function.
    """

    # Get the number of data points
    n = y.shape[0] if y.shape else 1

    # Create the error vector (i.e. yn - the predicted n-th value)
    e = y - tx.dot(w)
    

    return - 1 / n * tx.T.dot(e)

def compute_loss(y: np.ndarray, tx: np.ndarray, w: np.ndarray, lambda_ = 0, cf: str = "mse") -> float:
    """
    Calculate the loss using either MSE, RMSE or MAE for  linear.
    y: ndarray
        Array that contains the correct values to be predicted.
    tx: ndarray
        Matrix that contains the data points. The first column is made of 1s.
    w: ndarray
        Array containing the linear parameters to test.
    
    lambda_: float
        The regularization lambda.
    cf: str
        String indicating which cost function to use; "mse" (default), "rmse" or "mae".
    Returns
    -------
    loss: float
        The loss for the given linear parameters.
    """

    # Check whether the mode parameter is valid
    valid = ["mse", "rmse", "mae"]
    assert cf in valid, "Argument 'cf' must be either " + \
        ", ".join(f"'{x}'" for x in valid)

    # Create the error vector (i.e. yn - the predicted n-th value)
    e = y - tx.dot(w)
    
    # Compute the regularizer if it exists
    lambda_p = lambda_ * 2 * tx.shape[0] if lambda_ else 0

    if "mse" in cf:
        mse = e.T.dot(e) / (2 * len(e))
        if cf == "rmse":
            return math.sqrt(2 * mse) + lambda_
        return mse
    # mae
    return np.mean(np.abs(e)) + lambda_

def compute_logistic_gradient(y: np.ndarray, tx: np.ndarray, w: np.ndarray, lambda_: float = 0) -> np.ndarray:
    """"
    Calculates the of logistic linear loss.
    Parameters
    ----------
    y: ndarray
        Array that contains the correct values to be predicted.
    tx: ndarray
        Matrix that contains the data points. The first column is made of 1s.
    w: ndarray
        Array containing the linear parameters to test.
    lambda_: float
        The lambda used for regularization. Default behavior is without regularization.
    Returns
    -------
    gradient: np.ndarray
        The gradient for the given logistic linear parameters.
    """

    # Find the regularizer component (if lambda != 0)
    regularizer = lambda_ * w if lambda_ else 0

    return tx.T.dot(sigmoid(tx.dot(w)) - y) + regularizer

def compute_logistic_loss(y: np.ndarray, tx: np.ndarray, w: np.ndarray, lambda_: float = 0) -> float:
    """"
    Calculates the loss for logistic linear.
    Parameters
    ----------
    y: ndarray
        Array that contains the correct values to be predicted.
    tx: ndarray
        Matrix that contains the data points. The first column is made of 1s.
    w: ndarray
        Array containing the linear parameters to test.
    lambda_: float
        The lambda used for regularization. Default behavior is without regularization.
    Returns
    -------
    loss: float
        The loss for the given logistic linear parameters.
    """

    # Find the regularizer (if lambda != 0)
    regularizer = lambda_ / 2 * (np.linalg.norm(tx) ** 2) if lambda_ else 0

    summing = np.sum(np.log(1 + np.exp(tx.dot(w))))
    y_component = y.T.dot(tx.dot(w)).flatten().flatten()

    return summing - y_component + regularizer
    
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
