import pandas as pd
import numpy as np
import random

"""
    This file is a linear Support Vector Machine for binary classification.
    our goal
    -   min (1/2)||w||^2 s.t t_n(W^T phi(X_n) + b) >= 1 for every n

    How to use the class:
    ex)
        model = SGD_SVC()
        model.fit(xvalues, yvalues, c_value, k_value)
        result = model.predict(test_xvalues)
"""
class SGD_SVC:
    def __init__(self):
        """
            This function initialize the model.

            Parameter(s): None
            Returns: None
        """
        self.M = None
        self.X, self.T = None, None
        self.W, self.b = None, None

    def fit(self, xvals, yvals, C = 20, K = 50):
        """
            this function trains the model.

            Parameters:
                - xvals: X values
                - yvals: y values
                - C : penality for slack variable
                - K : Sample Size for stochastic gradient descent

            Returns: None
        """
        N, M = xvals.shape[0], xvals.shape[1] #dimension for the input data
        self.M = M
        self.X, self.T = xvals, yvals 
        X, T = self.X, self.T

        """
            Training the model based on the training data set
            and stochastic gradient descent,
            the performance depends on the initial b value.

            Initial mu value is 0.25 and descend by 1.015 at each training loop.
        """
        mu = 0.25 #learning rate
        self.W, self.b = np.zeros((M, 1), dtype = float), random.uniform(-1, 1)
        for _ in range(1000): #training loop - implemented stochastic gradient descent
            sample_X = X.sample(K)       #random sample of K inputs
            sample_T = T[sample_X.index] #corresponding ground truth
            new_sample_index = range(K)
            
            sample_X.index = sample_T.index = new_sample_index
            for i in new_sample_index:
                Tn, Xn = sample_T.iloc[i], np.asanyarray(sample_X.iloc[i]).reshape((M, 1))
                Yn = self.classify(Xn)
                isError = 1 - (Tn * Yn) > 0
                gradient_w = (self.W / K) - (C * Tn * Xn) if isError else self.W / K

                self.W -= (mu * gradient_w)
            mu /= 1.015

    def classify(self, Xn):
        """
            This function classifies based on the given x input.

            Parameter(s):
                - Xn : an X value

            Returns:
                - Yn : a classification result
        """
        return (np.dot(self.W.T, Xn) + self.b)[0][0] #classification result

    def predict(self, new_vals):
        """
            This function classifies based on the rows of input data.

            Parameter(s):
                - new_vals : test dataset
            
            Returns:
                - prediction results as a Panda Series.
        """
        index = np.arange(new_vals.shape[0])
        new_vals.index = index
        seriesToArr = lambda series : np.asanyarray(series).reshape((self.M, 1))

        data = []
        for i in index:
            result = 0
            classify_result = self.classify(seriesToArr(new_vals.iloc[i]))
            if classify_result > 0:
                result = 1
            elif classify_result < 0:
                result = -1
            data.append(result)

        return pd.Series(data = data, index = index)
        