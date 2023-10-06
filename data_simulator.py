from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd

from scipy.stats import chi2
from sklearn.model_selection import train_test_split


class DataSimulatorSimple:
    """Class to generate a simulated regression dataset with the following characteristics:
    - uncorrelated features
    - prespecified true and outlier beta coefficients
    - vertical outliers and/or bad leverage points
    """

    def __init__(
        self,
        n_observations: int = 1000,
        n_features: int = 10,
        contamination_distance_y: float = 1,
        contamination_distance_x: float = 1,
        bad_leverage_perc: float = 0,
    ):
        self.n_observations = n_observations
        self.n_features = n_features
        self.contamination_distance_y = contamination_distance_y
        self.contamination_distance_x = contamination_distance_x
        self.bad_leverage_perc = bad_leverage_perc
        self.mu = np.repeat(0, n_features)
        self.sigma = np.diag(np.repeat(1, n_features))
        self.beta_true = np.repeat(1, n_features)
        self.beta_contaminated = np.repeat(-1, n_features) * contamination_distance_y

    def get_data(
        self,
        test_perc: float = 0,
        outlier_perc: float = 0,
        random_seed: int = 42,
        return_outlier_idx: bool = False,
    ) -> Union[
        Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, Optional[np.ndarray]],
        Tuple[
            np.ndarray,
            Optional[np.ndarray],
            np.ndarray,
            Optional[np.ndarray],
            Dict[str, np.ndarray],
        ],
    ]:
        """Generate training and testing data

        Args:
            test_perc (optional): percentage of training data. Defaults to 0.
            outlier_perc (optional): percentage of contaminated data in the training set. Defaults to 0.
            random_seed (optional): control randomness. Defaults to 42.

        Returns:
            X_train, X_test, y_train, y_test
        """
        np.random.seed(random_seed)
        X = np.random.multivariate_normal(mean=self.mu, cov=self.sigma, size=self.n_observations)
        y = X @ self.beta_true.reshape(-1, 1) + np.random.randn(self.n_observations, 1)

        if test_perc > 0:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_perc, random_state=random_seed
            )
        else:
            X_train, y_train = X, y
            X_test = y_test = None

        n_contamination_samples = int(len(X_train) * outlier_perc)
        n_bad_leverage_points = int(self.bad_leverage_perc * n_contamination_samples)
        X_train[:n_bad_leverage_points] = (
            X_train[:n_bad_leverage_points] * self.contamination_distance_x
        )
        y_train[:n_contamination_samples] = X_train[
            :n_contamination_samples
        ] @ self.beta_contaminated.reshape(-1, 1) + np.random.randn(n_contamination_samples, 1)

        if return_outlier_idx:
            return (
                X_train,
                X_test,
                y_train,
                y_test,
                {
                    'bad_leverage': np.arange(0, n_bad_leverage_points),
                    'vertical': np.arange(n_bad_leverage_points, n_contamination_samples),
                },
            )
        return X_train, X_test, y_train, y_test


class DataSimulatorA09:
    """Class to generate a simulated regression dataset using the "A09" correlation matrix (see https://rdrr.io/cran/cellWise/man/generateCorMat.html)
    Outliers are only added for the training data, while the test set only contains uncontaminated data

    This implementation only allows a constant clean location (i.e. a vector of a repeated constant)
    and a constant regression coefficent vector (i.e. a vector of a repeated constant)
    """

    def __init__(
        self,
        n_observations: int = 1000,
        n_features: int = 10,
        rho: float = -0.9,
        noise_std: float = 1,
        mu: float = 0,
        contamination_distance: float = 10,
        contamination_spread: float = 0,
        regression_intercept: float = 1,
        regression_coef: float = 1,
        contamination_vertical_shift: float = -10,
        bad_leverage_perc: float = 0.5,
    ):
        """Create a data simulator

        Args:
            n_observations (int, optional): Number of obserbations. Defaults to 1000.
            n_features (int, optional): Number of features. Defaults to 10.
            rho (float, optional):
                Factor that determines the X data covariance matrix with the following formula:
                rho^|i-j|, where i and j are row and column indices. Defaults to -0.9.
            noise_std (float, optional):
                Standard deviation of the gaussion noise added to y
                (y = BX + noise(N(0, noise_std))). Defaults to 1.
            mu (float, optional): Clean location vector is a vector of this constant repeated n_features. Defaults to 0.
            contamination_distance (float, optional): dx, i.e. "horizontal" shift of outliers in the X_data. Defaults to 10.
            contamination_spread (float, optional):
                factor to multiply the clean covariance matrix with in order to generate outliers.
                If 0, all outliers are equal to the contaminated location vector. Defaults to 0.
            regression_intercept (float, optional): Intercept for the regression data. Defaults to 1.
            regression_coef (float, optional):
                Vector of coefficients for the regression is a vector of this constant repeated n_feature times. Defaults to 1.
            contamination_vertical_shift (float, optional): dy, "vertical" shift of contamination data. Defaults to -10.
        """
        self.n_observations = n_observations
        self.n_features = n_features
        self.rho = rho
        self.noise_std = noise_std
        self.contamination_distance = contamination_distance
        self.contamination_spread = contamination_spread
        self.contamination_vertical_shift = contamination_vertical_shift
        self.clean_location_vector = np.repeat(mu, repeats=n_features)
        self.regression_coefficients = np.concatenate(
            ([regression_intercept], np.repeat(regression_coef, repeats=n_features))
        )
        self.covariance_matrix = self._generate_covariance_matrix()
        self.contaminated_location_vector = self._get_contaminated_location_vector()
        self.bad_leverage_perc = bad_leverage_perc

    def get_data(
        self,
        test_perc: float = 0,
        outlier_perc: float = 0,
        random_seed: int = 42,
        return_outlier_idx: bool = False,
    ) -> Union[
        Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, Optional[np.ndarray]],
        Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, Optional[np.ndarray], np.ndarray],
    ]:
        np.random.seed(random_seed)
        X_train, X_test = self._generate_X_data(self.n_observations, clean=True), None
        y_train, y_test = self._get_y_data(X_train, clean=True), None
        if test_perc > 0:
            X_train, X_test, y_train, y_test = train_test_split(
                X_train, y_train, test_size=test_perc, random_state=random_seed
            )
        if outlier_perc > 0:
            n_total = X_train.shape[0]
            n_outliers = int(n_total * outlier_perc)
            n_bad_leverage = int(self.bad_leverage_perc * n_outliers)
            n_clean = n_total - n_outliers
            X_bad_leverage = self._generate_X_data(n_bad_leverage, clean=False)
            X_vertical_outliers = self._generate_X_data(n_outliers - n_bad_leverage, clean=True)
            X_contaminated = np.concatenate((X_bad_leverage, X_vertical_outliers))
            y_contaminated = self._get_y_data(X_contaminated, clean=False)
            clean_idx = np.random.choice(n_total, n_clean, replace=False)
            scrambled_idx = np.random.permutation(n_total)
            X_train = np.concatenate((X_contaminated, X_train[clean_idx]))[scrambled_idx]
            y_train = np.concatenate((y_contaminated, y_train[clean_idx]))[scrambled_idx]

            outlier_idx = np.argwhere(scrambled_idx < n_outliers).flatten()
        else:
            outlier_idx = np.array([])

        if return_outlier_idx:
            return X_train, X_test, y_train, y_test, outlier_idx
        return X_train, X_test, y_train, y_test

    def _generate_covariance_matrix(self) -> np.ndarray:
        """The A09 covariance matrix is defined as a matrix where the elements are equal to
        (\rho)^{|i-j|}, with rho a constant between [-1, 1], and i, j are row and columbn indices respectively
        """
        row_idx = np.arange(self.n_features).reshape(-1, 1)
        column_idx = np.arange(self.n_features).reshape(1, -1)
        return np.power(self.rho, np.abs(row_idx - column_idx))

    def _get_contaminated_location_vector(self) -> np.ndarray:
        """Get the a location vector which has a Mahalanobis distance
        to the clean location vector of a prespecified distance.
        Contamination is added in the direction of the 'smallest' eigenvector
        """
        if self.n_features == 1:
            return self.clean_location_vector + self.contamination_distance
        eigenvalues, eigenvectors = np.linalg.eig(self.covariance_matrix)
        selected_direction = eigenvectors[:, np.argmin(eigenvalues)]
        mahalanobis_distance = (
            (selected_direction - self.clean_location_vector).reshape(1, -1)
            @ np.linalg.inv(self.covariance_matrix)
            @ (selected_direction - self.clean_location_vector).reshape(-1, 1)
        )[0][0]
        scaled_direction = selected_direction / np.sqrt(mahalanobis_distance)
        return scaled_direction * np.sqrt(
            chi2.ppf(0.975, self.n_features - 1) * self.contamination_distance
        )

    def _generate_X_data(self, n_observations: int, clean: bool = True) -> np.ndarray:
        if clean:
            return np.random.multivariate_normal(
                mean=self.clean_location_vector, cov=self.covariance_matrix, size=n_observations
            )
        else:
            if self.contamination_spread == 0:
                return np.repeat(
                    self.contaminated_location_vector.reshape(1, -1), n_observations, axis=0
                )
            else:
                return np.random.multivariate_normal(
                    mean=self.contaminated_location_vector,
                    cov=self.covariance_matrix * self.contamination_spread,
                    size=n_observations,
                )

    def _get_y_data(self, X_data: np.ndarray, clean: bool = True) -> np.ndarray:
        n_observations = len(X_data)
        augmented_X = np.c_[np.ones(n_observations), X_data]
        y_data = augmented_X @ self.regression_coefficients.reshape(-1, 1)
        gaussian_noise = np.random.normal(loc=0, scale=self.noise_std, size=(n_observations, 1))
        if clean:
            return y_data + gaussian_noise
        else:
            return y_data + 0.01 * gaussian_noise + self.contamination_vertical_shift
