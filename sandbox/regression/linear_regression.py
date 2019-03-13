import os
import sys
import sklearn
import numpy as np
import matplotlib.pyplot as plt

# dataset import
path = str(os.path.abspath(os.path.dirname(__file__)))
dpath = '/'.join(path.split('/')[:-1])
if dpath not in sys.path[:5]:
    sys.path.insert(1, dpath)
from data import dataset



# Data
# ====
# Diabetes features: ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
# Diabetes target : progression of disease over 1y
diabetes = dataset.Diabetes()  # wrapper of sklearn.datasets.load_diabetes
N, D = diabetes.shape  # (442, 10)
num_test = 65 # approx 15%
diabetes.split_dataset(num_test)


#-----------------------------------------------------------------------------#
#                                Linear Model                                 #
#-----------------------------------------------------------------------------#

###  UNIVARIATE  ###
def least_squares_regression(x, y):
    """ compute coefficients of linear function with ordinary least squares

    To fit a line to the data, we find the weights (coeffs) that minimize the
    L2 loss
    """
    assert x.shape == y.shape
    n = x.shape[0]
    # sums
    x_sum  = x.sum()
    y_sum  = y.sum()
    xy_sum = (x*y).sum()
    x2_sum = np.square(x).sum()
    x_sum2 = np.square(x.sum())

    # weights
    w1 = (n*xy_sum - x_sum*y_sum) / (n*x2_sum - x_sum2)
    w0 = (y_sum - w1*x_sum) / n
    return w1, w0

def fit_line(x, w1, w0):
    """ regression over sample given coeffs W
    yhat = f(x) = w1*x + w0
    """
    return w1 * x + w0


if __name__ == '__main__':
    """ Sample plot over a feature using ordinary least squares regression """

    # Select feature of interest
    feature_idx  = 2
    feature_name = diabetes.feature_names[feature_idx]
    x_train, y_train, x_test, y_test = diabetes.feature_set(feature_idx)

    # regression over train, test
    W = least_squares_regression(x_train, y_train)
    yhat = fit_line(x_test, *W)

    # plot
    plt.scatter(x_test, y_test, c='b', label=f"{feature_name} : disease progression")
    plt.plot(x_test, yhat, c='r', label='regression line')
    plt.grid()
    plt.legend()
    plt.title(f'Univariate linear regression: {feature_name}')
    plt.xlabel(feature_name)
    plt.ylabel('diseason progression')
    plt.show()



""" Notes
Linear regression is a learning algorithm which learns
a model which is a linear combination of the input features.


Linear function
===============
A univariate linear function (a straight line) with
input x and output y has the form

    y = w1*x + w0

where w0 and w1 are real-valued coefficients to be learned.
We think of the coefficients as weights, as the value of
y is changed by changing the relative weight of one term
or another.

Linear regression
=================
We define W to be the vector [w0, w1], and define

    h_W(x) = w1*x + w0

The task of finding the h_W, the linear model, that best
fits the data is called **linear regression**.

Learning
========
To fit a line to the data, we find the values of W that
minimize the empirical loss. It is 'traditional' to
use the squared loss function, L2, summed over all the
training examples.†

Loss(h_W) = sum(L2(Y, h_W(X)))
          = sum(square(Y - h_W(X)))
          = sum(square(Y - (w1*X + w0)))


    †: Gauss showed that if the Y values have normally
       distributed noise, then the most likely values
       of W are obtained by minimizing the sum of the
       square of the errors.

Loss minimization
=================
We want to find W* = argmin_W(Loss(h_W)). h_W is minimized
when its partial derivatives wrt w0 and w1 are zero,
eg when d_Loss/d_w0 = 0 and d_Loss/d_w1 = 0
These equations have the unique solution:

    w1 = (N * sum(X * Y) - sum(X)*sum(Y)) / (N * sum(X**2) - sum(X)**2)
    w0 = (sum(Y) - w1*sum(X)) / N

For univariate linear regression, the weight space defined
by w0 and w1 is 2D, so graphing the loss as a function of
w0 and w1 gives us a 3D plot. Such a plot shows the loss
as convex, implying no local minima. This is true for
every linear regression problem with an L2 loss func.
This is a major constraint of a linear model.

Optimization
============
To go beyond linear models, we need to understand that
equations defining a min loss will often have no closed-form
solution--instead being a general optimization search problem
in a continuous weight space.

This problem can be addressed by hill-climbing algorithm
that follows the gradient of the function to be optimized.
Since we are minimizing the loss, we use gradient descent.

The partial derivatives for Loss wrt w0 and w1, for a
single training sample (x,y):

    gw0 = -2 * (y - h_W(x))
    gw1 = -2 * (y - h_W(x)) * x

For a batch of training samples, we take the derivative of
the sum, which is the sum of the individual derivatives.

For vanilla minibatch grad descent, with learning rate ⍺
(and the -2 scalar folded into ⍺):

    w0 ← w0 + ⍺ * sum_j(Yj - h_W(Xj))
    w1 ← w1 + ⍺ * sum_j(Yj - h_W(Xj)) * Xj

Multivariate linear regression
==============================
Where each sample x in X is a vector of n-elements.
Same thing as univariate, just with dot prods against
weight vecs.

    h_W(x) = w.(x) = w.T * x = sum(wi * xi)

Regularization
==============
While we need not worry about overfitting with univariate
linear regression, we do with multi.

With multivariate linear regression in high-dimensional
space, it is possible that some dim that is actually
irrelevant appears by chance to be useful.

Thus, it is common to use regularization on multivariate
linear funcs to avoid overfitting.

Total cost
----------
With regularization, we minimize the *total cost* of a
hypothesis, counting both the empirical loss and the
complexity of the hypothesis:

    Cost(h) = EmpLoss(h) + 𝝺*Complexity(h)

For linear funcs, the complexity can be specified as a func
of the weights. We can consider a family of regularization
functions:

    Complexity(h_W) = Lp(W) = sum(abs(w)**p)

L1 sparsity
-----------
Where p == 1 is the L1 regularizer, p == 2 --> L2. The
L1 is often preferred as it leads to a sparse model:
where many weights are set to zero, effectively declaring
the corresponding attributes to be irrelevant (just as
decision trees do).

Hypotheses that discard attributes can be easier for a
human to understand, and the may be less likely to overfit.

#-------------------------------------------------------------

NOTE ON LINEAR CLASSIFICATION:
    Linear funcs can also be used to do classification.
    In these problems, the linear function defines a
    *decision boundary*, which is a line (or surface,
    in higher dims) that separates the classes.

    A linear decision boundary is called a "linear separator"
    and data that admit such a separator are called
    **linearly separable**.†

        †: This is your answer to when you use more
           exotic/expensive learning models (eg, DNN):
           is the data linearly separable? eg,
           can you cleanly draw a line separating the
           classes?
"""

