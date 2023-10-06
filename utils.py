import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing._data import _handle_zeros_in_scale

from custom_wrapping import wrap, XYWrapRegressor, XWrapRegressor


class CustomRobustScaler(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.center_ = np.nanmedian(X, axis=0)
        self.scale_ = stats.median_absolute_deviation(X, axis=0)
        _handle_zeros_in_scale(self.scale_, copy=False)
        return self

    def transform(self, X):
        return (X - self.center_) / self.scale_

    def inverse_transform(self, X):
        return (X * self.scale_) + self.center_


def get_smart_initialisation_weights(
    X: np.ndarray, y: np.ndarray, alpha: float, rescale: bool = False
) -> np.ndarray:
    """Get the weight vector of an OLS applied on
    the subset of observations with the smallest mahalanobis distance
    using mean and covariance of the wrap transformed Xy data

    Args:
        - X: features
        - y: target
        - alpha: fraction of data to use in hsubset
        - rescale: whether to use the rescaled version of wrapping

    Returns:
         A vector of shape (n_features + 1, ) with OLS coefficients (bias at index 0)
    """

    ols = XYWrapRegressor(alpha=alpha, rescale=rescale).fit(X, y)

    return np.hstack((ols.model.intercept_, ols.model.coef_.flatten()))


def get_smart_initialisation_weights_alt(
    X: np.ndarray, y: np.ndarray, alpha: float, rescale: bool = False
) -> np.ndarray:
    """Get the weight vector of an HuberRegression applied on
    the subset of observations with the smallest mahalanobis distance
    using mean and covariance of the wrap transformed X data

    Args:
        - X: features
        - y: target
        - alpha: fraction of data to use in hsubset
        - rescale: whether to use the rescaled version of wrapping

    Returns:
         A vector of shape (n_features + 1, ) with OLS coefficients (bias at index 0)
    """
    model = XWrapRegressor(alpha=alpha, rescale=rescale).fit(X, y)

    return np.hstack((model.model.intercept_, model.model.coef_.flatten()))
