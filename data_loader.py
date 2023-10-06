from __future__ import annotations
import itertools
import numpy as np
import matplotlib.pyplot as plt

from typing import Callable, Iterable, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.base import TransformerMixin
from sklearn.datasets import load_svmlight_file


class DataLoader:
    """Class to load datasets or generate random data"""

    def __init__(
        self,
        datapath: Optional[str] = None,
        test_perc: float = 0.2,
        outlier_perc: float = 0,
        bad_leverage_perc: float = 0,  # percentage relative to outlier_perc
        n_features: int = 10,
        n_obs: int = 100,
        intercept: Optional[float] = None,
        mean: float = 0,
        variance: float = 1,
        noise_std: float = 1,
        random_state: int = 42,
        scaler: Optional[Callable[..., TransformerMixin]] = None,
        contamination_distance_y: float = 10,
        contamination_distance_x: float = 5,
    ):
        self.random_state = random_state
        self.test_perc = test_perc
        self.outlier_perc = outlier_perc
        self.bad_leverage_perc = bad_leverage_perc
        self.n_features = n_features
        self.n_obs = n_obs
        self.intercept = intercept
        self.mean = mean
        self.variance = variance
        self.noise_std = noise_std
        self.scaler = scaler
        self.contamination_distance_y = contamination_distance_y
        self.contamination_distance_x = contamination_distance_x
        self.outlier_idx: Iterable[int] = []
        self.vertical_idx: Iterable[int] = []
        self.leverage_idx: Iterable[int] = []
        if datapath:
            self.X, self.y = load_svmlight_file(datapath)
            self.X = self.X.toarray()
            self.y = self.y.reshape(-1, 1)
            self.n_obs, self.n_features = self.X.shape
        else:
            self.X, self.y = self._get_random_normal_data()

        if test_perc > 0:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=self.test_perc, random_state=self.random_state
            )
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = (
                self.X.copy(),
                None,
                self.y.copy(),
                None,
            )

        if self.scaler is not None:
            self.X_scaler = self.scaler().fit(self.X_train)
            self.y_scaler = self.scaler().fit(self.y_train)
            self.X_train = self.X_scaler.transform(self.X_train)
            self.y_train = self.y_scaler.transform(self.y_train)
            self.X_test = self.X_scaler.transform(self.X_test)
            self.y_test = self.y_scaler.transform(self.y_test)

        np.random.seed(self.random_state)
        self._add_outliers()

    def get_data(self) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, Optional[np.ndarray]]:
        """Return tuple training and test data
        Returns:
            X_train, X_test, y_train, y_test
        """
        return self.X_train, self.X_test, self.y_train, self.y_test

    def _add_outliers(self) -> None:
        if self.outlier_perc > 0:
            self.outlier_idx = np.random.choice(
                np.arange(0, self.X_train.shape[0]),
                int(self.outlier_perc * self.X_train.shape[0]),
                replace=False,
            )
            if self.bad_leverage_perc > 0:
                self.leverage_idx = np.random.choice(
                    self.outlier_idx,
                    int(self.bad_leverage_perc * len(self.outlier_idx)),
                    replace=False,
                )
                self.vertical_idx = [i for i in self.outlier_idx if i not in self.leverage_idx]
            else:
                self.vertical_idx = self.outlier_idx

            self.y_train[self.outlier_idx, :] += self.contamination_distance_y
            eigenvalues, eigenvectors = np.linalg.eig(np.cov(self.X_train.T))
            selected_direction = eigenvectors[:, np.argmin(eigenvalues)]
            self.X_train[self.leverage_idx, :] += selected_direction * self.contamination_distance_x

    def _get_random_normal_data(self) -> Tuple[np.ndarray, np.ndarray]:
        np.random.seed(self.random_state)
        X_data = np.random.multivariate_normal(
            mean=np.repeat(self.mean, self.n_features),
            cov=np.diag(np.repeat(self.variance, self.n_features)),
            size=self.n_obs,
        )
        self.true_beta = np.linspace(1, self.n_features, self.n_features).reshape((-1, 1))
        noise_eps = np.random.normal(scale=self.noise_std, size=(self.n_obs, 1))
        y_data = np.matmul(X_data, self.true_beta) + noise_eps
        if self.intercept:
            y_data += self.intercept
        return X_data, y_data

    def reshuffle_train_test(self, random_state: int = 1):
        self.random_state = random_state
        if self.test_perc > 0:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=self.test_perc, random_state=self.random_state
            )
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = (
                self.X.copy(),
                None,
                self.y.copy(),
                None,
            )

        np.random.seed(self.random_state)
        self._add_outliers()

    def plot_train_data(self, figsize=(20, 20)):
        X_train, X_test, y_train, y_test = self.get_data()
        n_feat = X_train.shape[1]

        fig = plt.figure(figsize=figsize)
        combos = itertools.product(range(n_feat + 1), range(n_feat + 1))
        mpl_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        clean_idx = [i for i in np.arange(len(X_train)) if i not in self.outlier_idx]
        for i, j in combos:
            ax = fig.add_subplot(n_feat + 1, n_feat + 1, i * (n_feat + 1) + j + 1)
            x1 = X_train[:, i] if i < n_feat else y_train
            x2 = X_train[:, j] if j < n_feat else y_train
            if i == j:
                ax.hist(x1[clean_idx], color=mpl_colors[0], alpha=0.7)
                ax.hist(x1[self.vertical_idx], color=mpl_colors[1], alpha=0.7)
                ax.hist(x1[self.leverage_idx], color=mpl_colors[2], alpha=0.7)
            else:
                _ = ax.scatter(
                    x1[clean_idx], x2[clean_idx], color=mpl_colors[0], alpha=0.5, label='clean'
                )
                _ = ax.scatter(
                    x1[self.vertical_idx],
                    x2[self.vertical_idx],
                    color=mpl_colors[1],
                    alpha=0.5,
                    label='vertical outlier',
                )
                _ = ax.scatter(
                    x1[self.leverage_idx],
                    x2[self.leverage_idx],
                    color=mpl_colors[2],
                    alpha=0.5,
                    label='bad leverage point',
                )
                points, labels = ax.get_legend_handles_labels()

            if i == 0:
                ax.set_title(f'X{j}' if j < n_feat else 'y')
            if j == 0:
                ax.set_ylabel(f'X{i}' if i < n_feat else 'y')

        fig.legend(
            points,
            labels,
            loc='upper center',
            bbox_to_anchor=(0.5, 0.95),
        )
        return fig
