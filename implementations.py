import numpy as np
from typing import Tuple

from random import randrange
from cost import compute_loss
from cost import calculate_mse
from gradient_decent import compute_gradient
from stochastic_gradient_descent import compute_stoch_gradient
from helpers import batch_iter


"""
def reg_logistic_regression(y: np.ndarray, tx: np.ndarray, lambda_: float,
                            initial_w: np.ndarray, max_iters: int, gamma: float) -> Tuple[np.ndarray, float]:
    
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
    

    return reg_logistic_regression(y, tx, 0, initial_w, max_iters, gamma)
"""


def least_squares(y: np.ndarray, tx: np.ndarray) -> Tuple[float, np.ndarray]:
    """Calculate the least squares solution.
       returns mse, and optimal weights.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.
    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """
    x = tx.T.dot(tx)
    y_ = tx.T.dot(y)
    w = np.linalg.solve(x, y)
    loss = compute_loss(y, tx, w)

    return loss, w


def ridge_regression(y: np.ndarray, tx: np.ndarray, lambda_: float) -> Tuple[float, np.ndarray]:
    """
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 0)
    array([ 0.21212121, -0.12121212])
    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 1)
    array([0.03947092, 0.00319628])
    """
    
    # Compute the lambda
    lambda_p = lambda_ * 2 * tx.shape[0]
    
    # Solve the linear system
    w = np.linalg.solve(tx.T.dot(tx) + lambda_p * np.eye(tx.shape[1]), tx.T.dot(y))
    
    # Compute the loss
    loss = compute_loss(y, tx, w, lambda_)
    
    return loss, w


def mean_squared_error_gd(y : np.ndarray, tx : np.ndarray, initial_w : np.ndarray, max_iters : int, gamma : float):
    """The Gradient Descent (GD) algorithm.
    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of GD
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    loss_gd = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, err = compute_gradient(y, tx, w)
        loss = calculate_mse(err)
        # gradient w by descent update
        w = w - gamma * grad
        # store w and loss
        ws.append(w)
        loss_gd.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return loss_gd, ws

def mean_squared_error_sgd(y : np.ndarray, tx : np.ndarray, initial_w : np.ndarray, batch_size, max_iters : int, gamma : float):
    """The Stochastic Gradient Descent algorithm (SGD).
    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize
    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    loss_sgd = []
    w = initial_w

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute a stochastic gradient and loss
            grad, _ = compute_stoch_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss
            loss = compute_loss(y, tx, w)
            # store w and loss
            ws.append(w)
            loss_sgd.append(loss)

        print("SGD({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return loss_sgd, ws