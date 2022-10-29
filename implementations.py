import numpy as np
import math
from typing import  Tuple
from random import randrange

from src.logistic import *
from src.gradient import *
from src.loss import *


ddef mean_squared_error_gd(y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray,
                     max_iters: int, gamma: float) -> Tuple[float, np.ndarray]:
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad = compute_gradient(y, tx, w)
        loss = compute_loss(y,tx,w)
        # gradient w by descent update
        w = w - gamma * grad
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws


def mean_squared_error_sgd(y: np.ndarray, tx: np.ndarray,initial_w: np.ndarray,
                     max_iters: int, gamma: float) -> Tuple[float, np.ndarray]:
    """Stochastic gradient descent."""
    # Define parameters to store w and loss
   
    ws = [initial_w]
    losses=[]
    w = initial_w
    
    for n_iter in range(max_iters):
        # compute loss, gradient
        rand_idx = np.random.randint(len(y))
        rand_tx = tx[rand_idx]
        rand_y = y[rand_idx]
        
        grad = compute_gradient(rand_y, rand_tx, w)  
        loss = compute_loss(y,tx,w)
        # gradient w by descent update
        w = w - gamma * grad
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("SGD({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws


def least_squares(y, tx):
    """
    Least squares regression using normal equations
    Args:
        y: labels
        tx: features
    Returns:
        w: optimized weight vector for the model
        loss: optimized final loss based on mean squared error
    """
    coefficient_matrix = tx.T.dot(tx)
    constant_vector = tx.T.dot(y)
    w = np.linalg.solve(coefficient_matrix, constant_vector)
    loss = compute_loss(y, tx, w)
    
    return loss, w


def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations
    Args:
        y: labels
        tx: features
        lambda_: regularization parameter
    Returns:
        w: optimized weight vector for the model
        loss: optimized final loss based on mean squared error
    """
    coefficient_matrix = tx.T.dot(tx) + 2 * len(y) * lambda_ * np.identity(tx.shape[1])
    constant_vector = tx.T.dot(y)
    w = np.linalg.solve(coefficient_matrix, constant_vector)
    loss = compute_loss(y, tx, w)
    return w, loss


def reg_logistic_regression(y: np.ndarray, tx: np.ndarray, lambda_: float,
                            initial_w: np.ndarray, max_iters: int, gamma: float) -> Tuple[np.ndarray, float]:
    """
    Does the regularized logistic linear.
    Parameters
    ----------
    y: ndarray
        Array that contains the correct values to be predicted.
    tx: ndarray
        Matrix that contains the data points. The first column is made of 1s.
    lambda_: float
        The lambda used for regularization. Default behavior is without regularization.
    initial_w: ndarray
        Array containing the linear parameters to start with.
    max_iters: int
        The maximum number of iterations to do.
    gamma: float
        Gradient descent stepsize
    Returns
    -------
    w: np.ndarray
        The linear parameters.
    loss: float
        The loss given w as parameters.
    """

    # init parameters
    threshold = 1e-8
    losses = []
    w = initial_w

    # start the logistic linear
    for iteration in range(max_iters):
        # get loss and update w.
        loss, gradient, w = gradient_descent_step(y, tx, w, gamma, lambda_, mode='logistic')
        # log info
        if iteration % 100 == 0:
            print("Current iteration={i}, loss={loss}".format(
                i=iteration, loss=loss))
            print("||d|| = {d}".format(d=np.linalg.norm(gradient)))
        # converge criterion
        losses.append(loss)
        # print("Current iteration={i}, loss={l}, ||d|| = {d}".format(i=iter, l=loss, d=np.linalg.norm(gradient)))
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    # visualization
    print("loss={l}".format(l=compute_loss(y, tx, w)))

    return w, losses[-1]


def logistic_regression(y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray,
                        max_iters: int, gamma: float) -> Tuple[np.ndarray, float]:
    """
    Computes the parameters for the logistic linear.
    Parameters
    ----------
    y: ndarray
        Array that contains the correct values to be predicted.
    tx: ndarray
        Matrix that contains the data points. The first column is made of 1s.
    initial_w: ndarray
        Array containing the linear parameters to start with.
    max_iters: int
        The maximum number of iterations to do.
    gamma: float
        Gradient descent stepsize
    Returns
    -------
    w: np.ndarray
        The linear parameters.
    loss: float
        The loss given w as parameters.
    """

    return reg_logistic_regression(y, tx, 0, initial_w, max_iters, gamma)
