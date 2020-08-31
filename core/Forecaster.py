"""
Forecaster
----------
Base class that defines an interface to the forecaster.
All other specific forecaster will inherit for this class.
"""
from builtins import object

class BaseForecaster(object):
    def fit(self, X, y):
        """
        train model
        :param X: feature matrix
        :param y: label vector
        """
        raise NotImplemented()

    def predict(self, X):
        """
        predict using the model
        :param X: feature matrix
        """
        raise NotImplemented()

    def score(self, X, y):
        """
        quantify the quality of the prediction. could be a metric score like
        rmse or it could be an error band around the prediction
        :param X: feature matrix (if needed)
        :param y: label vector (if needed)
        """
        raise NotImplemented()
