"""
Custom implementation of wrapping method proposed in Jakob Raymaekers & Peter J. Rousseeuw (2021) Fast Robust Correlation for High-Dimensional Data, Technometrics, 63:2, 184-198, DOI: 10.1080/00401706.2019.1677270
"""
from __future__ import annotations
import scipy.stats as ss
import logging
import numpy as np

from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.base import RegressorMixin, BaseEstimator


def wrap(
    X: np.ndarray,
    b: float = 1.5,
    c: float = 4.0,
    q1: float = 1.540793,
    q2: float = 0.8622731,
    rescale: bool = False,
) -> np.ndarray:
    """
    Implementation of wrapping using this transformation function:

    phi(z) = {
        z                                       if 0 <= |z| < b
        q1 * tanh(q2 * (c - |z|)) * sign(z)     if b <= |z| <= c
        0                                       if c <= |z|
    }

    Args:
        X: data to be transformed, must have shape (N, D)
        b: lower cutoff
        c: upper cutoff
        q1, q2: transformation parameters (see formula)
        rescale: whether to rescale the wrapped data so the robust location and scale
                 of the transformed data are the same as the original data

    Returns:
        transformed data
    """

    median = np.median(X, axis=0)
    mad = ss.median_abs_deviation(X, axis=0)
    mad_no_zero = np.where(mad == 0, 1, mad)

    z = (X - median) / mad_no_zero

    z_wrapped = np.where(
        np.abs(z) < b,
        z,
        np.where(np.abs(z) <= c, q1 * np.tanh(q2 * (c - np.abs(z))) * np.sign(z), 0),
    )
    if rescale:
        z_wrapped_mean = np.mean(z_wrapped, axis=0)
        z_wrapped_std = np.std(z_wrapped, axis=0)
        z_wrapped_std_no_zero = np.where(z_wrapped_std == 0, 1, z_wrapped_std)
        return (
            z_wrapped * (mad / z_wrapped_std_no_zero)
            + median
            - (z_wrapped_mean * (mad / z_wrapped_std_no_zero))
        )
    else:
        return z_wrapped * mad + median


def mahalanobis(X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """
    Calculate mahalonobis distances for a matric with a fiven mean and covariance
    Args:
        - X: (n, d) matrix with n observations and d variables
        - mean: (d,) shaped location vector)
        - cov: (d, d) shaped covariance matrix, must be invertible

    Returns:
        - a (n, 1) shaped vector of distances
    """
    differences = X - mean
    inv_cov = np.linalg.inv(cov)
    return np.array(
        list(
            map(
                lambda diff: np.sqrt(np.dot(np.dot(diff, inv_cov), diff.reshape(-1, 1))),
                differences,
            )
        )
    )


class XYWrapRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, alpha: float = 0.5, rescale: bool = False):
        """A regressor that fits an OLS on the h-subset with the
        smallest mahalnobis distances after applying the wrapping transformation
        on X and y data combined
        """
        self.alpha = alpha
        self.rescale = rescale
        self.logger = logging.getLogger('XYWrapRegressor')

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbosity: int = logging.INFO,
    ) -> XYWrapRegressor:

        self.logger.setLevel(verbosity)
        Xy = np.concatenate((X, y), axis=1)
        Xy_wrap = wrap(Xy, rescale=self.rescale)

        mean = Xy_wrap.mean(axis=0)
        # np.cov expects each row to represent a variable by default,
        # so rowvar=False is needed (alternative is to transpose)
        cov = np.cov(Xy_wrap, rowvar=False)

        distances = mahalanobis(Xy, mean, cov).flatten()

        h_subset = np.argsort(distances)[: int(self.alpha * len(Xy))]

        self.logger.debug('fitting linear regression on h-subset')

        self.model = LinearRegression().fit(X[h_subset], y[h_subset])

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class XWrapRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, alpha: float = 0.5, rescale: bool = False):
        """A regressor that fits a HuberRegression on the h-subset with the
        smallest mahalnobis distances after applying the wrapping transformation
        no X data only combined
        """
        self.alpha = alpha
        self.rescale = rescale
        self.logger = logging.getLogger('XWrapRegressor')

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbosity: int = logging.INFO,
    ) -> XWrapRegressor:
        self.logger.setLevel(verbosity)
        X_wrap = wrap(X, rescale=self.rescale)

        mean = X_wrap.mean(axis=0)
        # np.cov expects each row to represent a variable by default,
        # so rowvar=False is needed (alternative is to transpose)
        cov = np.cov(X_wrap, rowvar=False)

        distances = mahalanobis(X, mean, cov).flatten()

        h_subset = np.argsort(distances)[: int(self.alpha * len(X))]

        self.logger.debug('fitting HuberRegressor in hsubset')

        self.model = HuberRegressor(max_iter=200).fit(X[h_subset], y[h_subset].flatten())

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
