""" Code for experiments with a 'soft' version of LTS based on soft ranking"""
from __future__ import annotations
import logging
import numpy as np

from typing import Callable, Tuple, Optional, List, Union
from tqdm import tqdm
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.linear_model import LinearRegression

from scipy.optimize import minimize

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer, Dense
from tensorflow.python.framework.ops import EagerTensor
from tensorflow.python.training.optimizer import Optimizer


from fast_soft_sort import numpy_ops, tf_ops

TFOptimizer = Union[Optimizer]


class FastLTS(RegressorMixin, BaseEstimator):
    """
    Implementation of FAST-LTS model based on R implementation of the ltsReg method in the robustbase R package
    (cfr. https://www.rdocumentation.org/packages/robustbase/versions/0.93-8/topics/ltsReg)
    and the python implementation `Reweighted-FastLTS`
    (cfr. https://github.com/GiuseppeCannata/Reweighted-FastLTS/blob/master/Reweighted_FastLTS.py)
    """

    def __init__(
        self,
        alpha: float = 0.5,
        n_initial_subsets: int = 500,
        n_initial_c_steps: int = 2,
        n_best_models: int = 10,
        tolerance: float = 1e-15,
        random_state: int = 42,
    ):
        """Initialize a FAST LTS regressor

        Args:
            alpha (float): percentage of data to consider as subset for calculating the trimmed squared error.
                           Must be between 0.5 and 1, with 1 being equal to normal LS regression. Defaults to 0.5.
            n_initial_subset (int): number of initial subsets to apply C-steps on (cfr `m` in original R implementatino). Defaults to 500.
            n_initial_c_steps (int): number of c-steps to apply on n_initial_subsets before final c-steps until convergenge . Defaults to 2.
            n_best_models (int): number of best models after initial c-steps to consider until convergence. Defaults to 10.
            tolerance (float): Acceptable delta in loss value between C-steps. If current loss  -  previous loss <= tolerance, model is converged. Defaults to 1e-15.
        """
        self.alpha = alpha
        self.n_initial_subsets = n_initial_subsets
        self.n_initial_c_steps = n_initial_c_steps
        self.n_best_models = n_best_models
        self.tolerance = tolerance
        self.random_state = random_state
        self.logger = logging.getLogger("fast_lts")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        initial_weights: Optional[np.ndarray] = None,
        verbosity: int = logging.INFO,
    ):
        """Fit the model to the data

        Args:
            X (np.ndarray): Training features
            y (np.ndarray): Training labels
            initial_weights (Optional[np.ndarray], optional): Optionally pass fixed initial weights,
                            in case of n_initial_subsets > 1, this means all models start from the same initial weights.
                            There is therefore no benefit from setting n_initial_subsets > 1
                            Defaults to None.
            verbosity (int, optional): [description]. Defaults to logging.INFO.

        Returns:
            [type]: [description]
        """
        self.logger.setLevel(verbosity)
        h = int(X.shape[0] * self.alpha)
        self.logger.info(
            f"Applying {self.n_initial_c_steps} initial c-steps "
            f"on {self.n_initial_subsets} initial subsets"
        )
        y = y.reshape(-1, 1)
        lr_models, losses, h_subsets = self._apply_initial_C_steps(
            X, y, h, initial_weights=initial_weights, verbosity=verbosity
        )
        best_model_idxs = np.argsort(losses)[: self.n_best_models]
        best_model, best_loss, best_h_subset = (
            lr_models[best_model_idxs[0]],
            losses[best_model_idxs[0]],
            h_subsets[best_model_idxs[0]],
        )
        self.logger.info(f"Performing final C-steps on {self.n_best_models} best models")
        for model_idx in tqdm(best_model_idxs, disable=verbosity > logging.INFO):
            (
                current_model,
                current_h_subset,
                current_loss,
                _,
            ) = self._apply_C_steps_untill_convergence(
                lr_models[model_idx], losses[model_idx], X, y, h, self.tolerance, self.logger
            )

            if current_loss < best_loss:
                best_loss = current_loss
                best_model = current_model
                best_h_subset = current_h_subset
        self.model = best_model
        self.best_h_subset = best_h_subset

        return self

    def predict(self, X: np.ndarray):
        return self.model.predict(X)

    def _apply_initial_C_steps(
        self,
        X: np.ndarray,
        y: np.ndarray,
        h: int,
        initial_weights: Optional[np.ndarray],
        verbosity: int = logging.DEBUG,
    ) -> Tuple[List[LinearRegression], List[float], List[np.ndarray]]:
        """
        Perform initial c_steps on n_initial_subsets of size n_features + 1

        Returns:
            List of models, List of losses and List of h subsets
        """
        np.random.seed(self.random_state)
        lr_models = []
        losses = []
        h_subsets = []
        for seed, _ in tqdm(
            enumerate(range(self.n_initial_subsets), start=self.random_state),
            disable=verbosity > logging.INFO,
        ):
            lr_model = self._get_initial_model(X, y, seed, logger=self.logger)
            if initial_weights is not None:
                self.logger.warning(
                    f"Initializing models with fixed weights {initial_weights} instead of random initializations."
                )
                lr_model.intercept_ = initial_weights[[0]]
                lr_model.coef_ = initial_weights[None, 1:]
            h_subset_idx = self._get_h_subset(lr_model, X, y, h)
            for _ in range(self.n_initial_c_steps):
                h_subset_idx, lr_model = self._apply_C_step(lr_model, X, y, h)
            # get final residuals
            losses.append(self._get_loss_value(X, y, h_subset_idx, lr_model))
            lr_models.append(lr_model)
            h_subsets.append(h_subset_idx)
        return lr_models, losses, h_subsets

    @staticmethod
    def _get_initial_model(
        X: np.ndarray,
        y: np.ndarray,
        random_state: int = 42,
        logger: Optional[logging.Logger] = logging.getLogger("fast_lts"),
    ) -> LinearRegression:
        """Get a Linear Regression model that is fitted on
        a random subset of the data of size n_features + 1

        Args:
            X (np.ndarray): Feature data
            y (np.ndarray): Labels
            random_state (int, optional): Random seed, will determine the random subset. Defaults to 42.
            logger (Optional[logging.Logger], optional): logger object. Defaults to logging.getLogger('fast_lts').

        Returns:
            lr_model: A Linear Regression model fitted on a random subset
        """
        logger.debug(f"Getting initial model with seed {random_state}")
        n_obs, n_features = X.shape  # n, p
        np.random.seed(random_state)
        subset_idx = np.random.choice(n_obs, n_features + 1, replace=False)
        lr_model = LinearRegression().fit(X[subset_idx], y[subset_idx])
        return lr_model

    @staticmethod
    def _get_loss_value(
        X: np.ndarray,
        y: np.ndarray,
        h_subset_idx: Union[np.ndarray, list[int]],
        model: LinearRegression,
    ) -> float:
        """Get the Least Trimmed Squared loss for a specific model and h subset

        Args:
            X (np.ndarray): Features
            y (np.ndarray): Labels
            h_subset_idx (np.ndarray): Indices of h subset
            model (LinearRegression): A trained Linear Regression model

        Returns:
            mean squared residual of h_subset
        """
        y_true = y[h_subset_idx]
        y_pred = model.predict(X[h_subset_idx]).reshape(-1, 1)
        assert y_true.shape == y_pred.shape
        residuals = y_true - y_pred
        return np.sum(np.power(residuals, 2)) / len(h_subset_idx)

    @staticmethod
    def _apply_C_steps_untill_convergence(
        current_model: LinearRegression,
        previous_loss: float,
        X: np.ndarray,
        y: np.ndarray,
        h: int,
        tolerance: float = 1e-15,
        logger: logging.Logger = logging.getLogger("fast_lts"),
    ) -> Tuple[LinearRegression, np.ndarray, float, int]:
        """Apply c-steps until convergence

        Args:
            current_model (LinearRegression): model to start from
            previous_loss (float): reference loss value
            X (np.ndarray): Training data features
            y (np.ndarray): Training data targets
            h (int): Number of samples to consider in subset
            tolerance (float, optional): min delta in loss between iterations. Defaults to 1e-15.
            logger (logging.Logger, optional): logger. Defaults to logging.getLogger('fast_lts').

        Returns:
            Tuple[LinearRegression, np.ndarray, float, int]: [description]
        """
        iteration = 0
        while True:
            current_h_subset, current_model = FastLTS._apply_C_step(current_model, X, y, h)
            current_loss = FastLTS._get_loss_value(X, y, current_h_subset, current_model)
            logger.debug(
                f"Iteration {iteration}: current loss = {current_loss:.3f}, "
                f"previous loss = {previous_loss:.3f}"
            )
            if (previous_loss - current_loss) <= tolerance:
                break
            previous_loss = current_loss
            iteration += 1
        return current_model, current_h_subset, current_loss, iteration

    @staticmethod
    def _get_h_subset(
        lr_model: LinearRegression, X: np.ndarray, y: np.ndarray, h: int
    ) -> np.ndarray:
        """Get the indices of the h observations with the smallest residuals for a given model

        Args:
            lr_model (LinearRegression): A fitted Linear Regression Model
            X (np.ndarray): Features
            y (np.ndarray): Labels
            h (int): Number of observations to include in the subset

        Returns:
            np.ndarray: Array of indices for the h subset
        """
        residuals = y - lr_model.predict(X).reshape(-1, 1)
        return np.argsort(np.abs(residuals).flatten())[:h]

    @staticmethod
    def _apply_C_step(
        lr_model: LinearRegression, X: np.ndarray, y: np.ndarray, h: int
    ) -> Tuple[np.ndarray, LinearRegression]:
        """
        Apply a single C-step

        Returns:
            h subset indices, fitted lr model
        """
        h_subset_idx = FastLTS._get_h_subset(lr_model, X, y, h)
        lr_model = LinearRegression().fit(X[h_subset_idx], y[h_subset_idx])
        return h_subset_idx, lr_model


class ClassicLMS(RegressorMixin, BaseEstimator):
    """
    Implementation of LMS model based on the PROGRESS implementation
    (cfr. https://wis.kuleuven.be/stat/robust/papers/1997/progress.pdf)
    """

    def __init__(self, nr_of_subsamples: int = 3000, random_state: int = 42):
        """
        Args:
            nr_of_subsamples (optional): nr of subset of length p to draw. Defaults to 3000.
            random_state (optional): control random samples. Defaults to 42.
        """
        self.nr_of_subsamples = nr_of_subsamples
        self.random_state = random_state
        self.logger = logging.getLogger("LMS")

    def fit(self, X: np.ndarray, y: np.ndarray, verbosity: int = logging.INFO):
        """Find the regression with minimal median squared residual on the training set

        Args:
            X: Features
            y: Targets
            verbosity (optional): logging level. Defaults to logging.INFO.
        """
        self.logger.setLevel(verbosity)
        np.random.seed(self.random_state)
        self.results = []
        for i in range(self.nr_of_subsamples):
            if i % 1000 == 0:
                self.logger.debug(f"Processing sample {i} of {self.nr_of_subsamples}")
            subsample_idx = np.random.choice(np.arange(0, len(X)), X.shape[1], replace=False)
            lr = LinearRegression()
            lr.fit(X[subsample_idx], y[subsample_idx])
            self.results.append(
                {"model": lr, "median_squared_residual": self._get_loss_value(X, y, lr)}
            )
        best_result = min(self.results, key=lambda x: x["median_squared_residual"])
        self.model = best_result["model"]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    @staticmethod
    def _get_loss_value(X: np.ndarray, y: np.ndarray, model: LinearRegression):
        return np.median(np.power(y - model.predict(X), 2))


class SoftLTS(RegressorMixin, BaseEstimator):
    """
    Implementation of LTS using the soft sorting operator as defined in the paper
    `Fast Differentiable Sorting and Ranking Mathieu Blondel, Olivier Teboul, Quentin Berthet, Josip Djolonga In proceedings of ICML 2020 arXiv:2002.08871`
    using the python implementation from `https://github.com/google-research/fast-soft-sort`
    """

    def __init__(
        self,
        alpha: float = 0.5,
        regularization: str = "l2",
        regularization_strength: float = 1.0,
        random_state: int = 42,
        n_initial_subsets: int = 1,
        n_initial_iters: int = 1,
        n_best_models: int = 1,
    ):
        assert n_initial_subsets >= n_best_models, (
            f"`n_initial_subsets` must be greater than or equal to `n_best_models`, "
            f"but were {n_initial_subsets} and {n_best_models} respectively"
        )

        self.alpha = alpha
        self.regularization = regularization  # options: ('l2', 'kl') (quadratic or entropic)
        self.regularization_strength = regularization_strength  # epsilon
        self.random_state = random_state
        self.n_initial_subsets = n_initial_subsets
        self.n_initial_iters = n_initial_iters
        self.n_best_models = n_best_models
        self.logger = logging.getLogger("soft_lts")

    def _single_optimisation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seed: int = 42,
        initial_weights: Optional[np.ndarray] = None,
        calc_gradients: bool = True,
        max_iter: int = 300,
        ftol: float = 1e-15,
        verbosity: int = logging.INFO,
        convergence_warning: bool = False,
        normalize_gradients: bool = False,
    ) -> dict:
        """Optimisation until convergence or max_iter using the BFGS algorithm"""
        result = {}
        self.logger.debug("Loading initial weights and augmenting input")
        weights, X_augm = self._get_initial_weights_and_augment_input(X, y, seed)
        if initial_weights is not None:
            self.logger.warning(
                f"Setting initial weights to {initial_weights} instead of randomly initializing"
            )
            weights = initial_weights
        h = int(self.alpha * X.shape[0])
        self.logger.debug("Starting optimization")
        if calc_gradients:
            args = (
                X_augm,
                y,
                h,
                self.regularization_strength,
                self.regularization,
                True,
                normalize_gradients,
                self.logger,
            )
            jac = True
        else:
            args = (
                X_augm,
                y,
                h,
                self.regularization_strength,
                self.regularization,
                False,
                normalize_gradients,
                self.logger,
            )
            jac = None

        result["bfgs_result"] = minimize(
            fun=self._compute_soft_loss_and_gradient,
            x0=weights,
            jac=jac,
            args=args,
            method="L-BFGS-B",
            options={"maxiter": max_iter, "ftol": ftol, "disp": (verbosity <= logging.DEBUG)},
        )
        self.logger.debug("Finished optimization")
        if convergence_warning and not result["bfgs_result"].success:
            self.logger.warning(f"Optimizer did not converge")

        result["weights"] = result["bfgs_result"].x
        result["lr_model"] = LinearRegression()
        result["lr_model"].intercept_ = result["weights"][0]
        result["lr_model"].coef_ = result["weights"][1:]
        result["best_h_subset"] = np.argsort(np.abs(y.flatten() - result["lr_model"].predict(X)))[
            :h
        ]
        result["train_loss"] = FastLTS._get_loss_value(
            X, y, result["best_h_subset"], result["lr_model"]
        )

        return result

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        calc_gradients: bool = True,
        max_iter: int = 300,
        ftol: float = 1e-15,
        apply_c_steps: bool = False,
        initial_weights: Optional[np.ndarray] = None,
        normalize_gradients: bool = False,
        verbosity: int = logging.INFO,
    ):
        """Fit a softLTS using the BFGS optimization algorithm.

        Args:
            X (np.ndarray): Training features
            y (np.ndarray): Training labels
            calc_gradients (bool, optional): Whether to analytically calculate gradients or approximate using finite differences. Defaults to True.
            max_iter (int, optional): Max number of BFGS iterations. Defaults to 300.
            ftol (float, optional): tolerence parameter for BFGS algorithm (min improvement in loss between iterations). Defaults to 1e-15.
            apply_c_steps (bool, optional): whether to apply c-steps till convergence after the BFGS algorithm (cfr FastLTS). Defaults to False.
            initial_weights (Optional[np.ndarray], optional): Optionally specify a predefined set of initial weights (including intercept at index 0). Defaults to None.
            nornalize_gradients: whether to divide the soft gradients by their L2 norm
            verbosity (int, optional): logging level. Defaults to logging.INFO.
        """
        self.logger.setLevel(verbosity)
        self.logger.info(f"Optimizing weights for {self.n_initial_subsets} subsets")
        self.results = []
        for seed in tqdm(
            range(
                self.random_state * 10,
                self.random_state * 10 + self.n_initial_subsets * self.random_state,
                self.random_state,
            ),
            disable=verbosity > logging.INFO,
        ):
            result = self._single_optimisation(
                X=X,
                y=y,
                seed=seed,
                initial_weights=initial_weights,
                calc_gradients=calc_gradients,
                max_iter=self.n_initial_iters,
                ftol=ftol,
                verbosity=verbosity,
                convergence_warning=False,
                normalize_gradients=normalize_gradients,
            )
            self.results.append(result)

        self.logger.info(f"Finished optimization for {self.n_initial_subsets} initial subsets.")
        best_results = sorted(self.results, key=lambda x: x["train_loss"])[: self.n_best_models]
        self.final_models = []
        for i, result in enumerate(best_results, start=1):
            self.logger.debug(
                f"Optimising until convergence for model {i} out of {self.n_best_models}"
            )
            self.final_models.append(
                self._single_optimisation(
                    X=X,
                    y=y,
                    seed=i,
                    initial_weights=result["weights"],
                    calc_gradients=calc_gradients,
                    max_iter=max_iter,
                    ftol=ftol,
                    verbosity=verbosity,
                    convergence_warning=True,
                    normalize_gradients=normalize_gradients,
                )
            )

        best_result = min(self.final_models, key=lambda x: x["train_loss"])
        self.bfgs_result = best_result["bfgs_result"]
        self.model = best_result["lr_model"]
        self.best_h_subset = best_result["best_h_subset"]
        self.train_loss = best_result["train_loss"]

        if apply_c_steps:
            # TODO: consider applying c-steps to each of the n_best_models
            self.logger.info("Applying c-steps till convergence")
            (
                self.model,
                self.best_h_subset,
                self.train_loss,
                self.n_c_steps,
            ) = FastLTS._apply_C_steps_untill_convergence(
                current_model=self.model,
                previous_loss=self.train_loss,
                X=X,
                y=y,
                h=int(self.alpha * len(X)),
                logger=self.logger,
            )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    @staticmethod
    def _get_initial_weights_and_augment_input(
        X: np.ndarray, y: np.ndarray, random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        lr_model = FastLTS._get_initial_model(X, y, random_state)
        weights = np.hstack((lr_model.intercept_, lr_model.coef_.flatten()))

        return weights, np.append(np.ones((X.shape[0], 1)), X, axis=1)

    @staticmethod
    def _compute_soft_loss_and_gradient(
        weights: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        h: int,
        eps: float = 1.0,
        regul: str = "l2",
        return_gradients: bool = True,
        normalize_gradients: bool = False,
        logger: logging.Logger = None,
    ) -> Union[float, Tuple[float, np.ndarray]]:
        """Given linear regression coefficients, input data and soft sort parameters,
        calculate the sum of the `h` minumum soft squared residuals as well as the gradient with respect to the weights

        Args:
            - weights (ndarray): 1d array of shape (n_features + 1,), the first element is expected to be the intercept
            - X (ndarray): 2d array of shape (n_obs, n_features + 1) , the first element is expected to contain ones
            - y (ndarray): 1d array of shape (n_obs, )
            - h (int): number of samples to consider for calculating the loss and gradients (alpha x n_obs)
            - eps (float): regularization strength for soft sort. Default = 1.0
            - regul (strt): type of regularization for soft sort, either `l2` or `kl`. Default = 'l2'
        """
        if not logger:
            logger = logging.getLogger("soft_lts")
            logger.setLevel(logging.INFO)

        # start = time.time()
        W = weights.reshape((-1, 1))

        y_pred = np.matmul(X, W)  # shape = (n_input, 1)
        residuals = y_pred - y.reshape(-1, 1)  # shape = (n_input, 1)
        squared_residuals = residuals**2  # shape = (n_input, 1)

        # soft_sort residuals
        softsort = numpy_ops.SoftSort(
            values=squared_residuals.flatten(),
            direction="ASCENDING",
            regularization_strength=eps,
            regularization=regul,
        )

        soft_squared_residuals = softsort.compute()  # shape = (n_input, )
        soft_loss = np.sum(soft_squared_residuals[:h]) / h
        # print(soft_loss)
        # intermediate = time.time()
        # logger.debug(f"Calculating loss took {(intermediate - start):.0f} seconds")
        if not return_gradients:
            return soft_loss

        # d(soft_loss)/dW = d(soft_loss)/d(loss_VECTOR) * d(loss_VECTOR)/dw
        # the jacobian of softsort is a matrix of shape (n_output, n_input),
        # i.e. the first row contains the gradient of output 1 with respect to all inputs
        # For regular linear regression we consider the gradient of the SUM of squared losses
        # w.r.t. to the weights, i.e. de gradient of scalar function w.r.t. a vector
        # in this case we require the gradient of the VECTOR of squared residuals w.r.t. to the weights
        # i.e. we require a matrix of shape (n_inputs  x n_feature + 1)
        # softsort.jvp can only output a 1d vector, i.e. it expects a vector of shape (n_inputs, 1)
        # as input so in order the get the (n_inputs x n_features + 1) output that is required,
        # we need to loop over n_features + 1 dimensions
        # alternatively, we can use the jacobian directly, which also allows us the select the top h at once
        gradient_of_loss_vector_to_weights = 2 * residuals * X  # shape = (n_input x n_feat + 1)
        # the .jacobian() method is very slow, a custom variant of the built-in .jvp method is used
        # soft_gradients = np.matmul(softsort.jacobian()[:h, :],
        #                            gradient_of_loss_vector_to_weights
        #                            ).sum(axis=0) # shape = (n_feat + 1, )
        soft_gradients = (
            jacobian_matrix_mult(softsort, gradient_of_loss_vector_to_weights, max_idx=h)[
                :h, :
            ].sum(axis=0)
            / h
        )  # shape = (n_features + 1, )
        # logger.debug(f"Calculating gradient took {(time.time() - intermediate):.0f} seconds")
        if normalize_gradients:
            soft_gradients /= np.linalg.norm(soft_gradients)

        return soft_loss, soft_gradients


def jacobian_matrix_mult(
    ss: numpy_ops.SoftSort, matrix: np.ndarray, max_idx: Optional[int] = None
) -> np.ndarray:
    """Matrix version of built-in `jvp` (jacobian vector product) method.
    Works ~8 times faster than ss.jacobian() x matrix

    Args:
        ss (SoftSort): A precomputed SoftSort instance
        matrix (np.ndarray): RHS matrix to multiply the jacobian with

    Returns:
        np.ndarray: Result of matrix multiplication J x M

    Raises NotImplementedError:
        only `l2` regularization is supported a.t.m.
    """
    ss._check_computed()
    matrix = matrix[ss.permutation_, :]
    n_outputs = ss.isotonic_.solution_.shape[0]
    max_idx = n_outputs if max_idx is None else min(n_outputs, max_idx)
    start = 0
    return_value = np.zeros(shape=(max_idx, matrix.shape[-1]))
    for size in numpy_ops._partition(ss.isotonic_.solution_):
        end = start + size
        if ss.regularization == "l2":
            val = np.mean(matrix[start:end, :], axis=0)
        else:
            raise NotImplementedError("Only l2 regularization is support for this method a.t.m.")
        return_value[start:end, :] = val
        start = end
        if start >= max_idx:
            break
    return return_value


def get_trimmed_loss_value(model: Union[FastLTS, SoftLTS], X: np.ndarray, y: np.ndarray) -> float:
    """Scorer function for scikit-learn cross validation
    Returns the negative of the loss function because GridSearch will maximize this value
    """
    residuals = y.flatten() - model.predict(X).flatten()
    h = int(X.shape[0] * model.alpha)
    residuals = np.sort(residuals)[:h]
    return -np.sum(np.power(residuals, 2)) / h


def lts_loss_function(
    y_true: np.ndarray, y_pred: EagerTensor, h: int, epsilon: float = 1.0
) -> EagerTensor:
    """Tensorflow version of trimmed loss function"""
    squared_residuals = tf.square(y_true.reshape(-1, 1) - tf.reshape(y_pred, [-1, 1]))
    sorted_residuals = tf.reshape(
        tf_ops.soft_sort(tf.reshape(squared_residuals, [1, -1]), regularization_strength=epsilon),
        [-1, 1],
    )
    return tf.reduce_mean(sorted_residuals[:h], axis=0)


class PickableSequential(keras.Sequential):
    """Make keras Sequential model pickable (required for sklearn GridSearch with n_jobs > 1)"""

    def __getstate__(self):
        state = super().__getstate__()
        state.pop("_trackable_saver")
        state.pop("_compiled_trainable_state")
        return state


class SoftLTSSGD(RegressorMixin, BaseEstimator):
    """
    Variation of LTS using the soft sorting operator as defined in the paper
    `Fast Differentiable Sorting and Ranking Mathieu Blondel, Olivier Teboul, Quentin Berthet, Josip Djolonga In proceedings of ICML 2020 arXiv:2002.08871`
    using the python implementation from `https://github.com/google-research/fast-soft-sort`

    This implementation uses a Gradient Descent based optimizer in Tensorflow as opposed to the BFGS-optimization used in `SoftLTS`
    """

    def __init__(
        self,
        alpha: float = 0.5,
        regularization: str = "l2",
        regularization_strength: float = 1.0,
        learning_rate: float = 0.1,
        optimizer: Callable[..., TFOptimizer] = tf.optimizers.Adam,
        n_initial_subsets: int = 1,
        n_initial_iters: int = 1,
        n_best_models: int = 1,
        use_fast_lts_initialisation: bool = False,
        random_state: int = 42,
    ):
        self.alpha = alpha
        self.regularization = regularization  # options: ('l2', 'kl') (quadratic or entropic)
        self.regularization_strength = regularization_strength  # epsilon
        self.random_state = random_state
        self.tf_model = PickableSequential([Dense(units=1)])
        self.learning_rate = learning_rate
        self.optimizer = optimizer(learning_rate=learning_rate)
        self.n_initial_subsets = n_initial_subsets
        self.n_initial_iters = n_initial_iters
        self.n_best_models = n_best_models
        self.use_fast_lts_initialisation = use_fast_lts_initialisation
        self.logger = logging.getLogger("soft_lts_sgd")

    def _train_step(self, X: np.ndarray, y: np.ndarray, tf_model: PickableSequential) -> float:
        with tf.GradientTape() as tape:
            y_pred = tf_model(X, training=True)
            loss_value = lts_loss_function(
                y, y_pred, h=int(len(X) * self.alpha), epsilon=self.regularization_strength
            )

        grads = tape.gradient(loss_value, tf_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, tf_model.trainable_variables))

        return loss_value.numpy().mean()

    def _train(
        self,
        epochs: int,
        X: np.ndarray,
        y: np.ndarray,
        tf_model: Optional[PickableSequential] = None,
        history: Optional[List[float]] = None,
        patience: Optional[int] = None,
        tolerance: float = 1e-15,
    ) -> List[float]:
        patience = patience if not patience is None else epochs
        if tf_model is None:
            tf_model = self.tf_model
        history = [] if history is None else history
        counter = 0
        best_loss = 1e5
        # TODO: add early stopping
        pbar = tqdm(range(epochs), disable=self.logger.level > logging.DEBUG)
        for i in pbar:
            loss = self._train_step(X, y, tf_model)
            history.append(loss)
            pbar.set_postfix({"Train loss": loss})
            if (best_loss - loss) / max([abs(best_loss), abs(loss), 1]) > tolerance:
                counter = 0
            else:
                counter += 1
            if loss < best_loss:
                best_loss = loss
            if counter > patience:
                self.logger.warning(
                    f"Stopping early after {i} iterations. "
                    f"No loss improvements for the last {counter} iterations."
                )
                break

        return history

    def _single_optimisation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seed: int,
        initial_weights: Optional[np.ndarray] = None,
        epochs: int = 100,
        tf_model: Optional[PickableSequential] = None,
        history: Optional[List[float]] = None,
        patience: Optional[int] = None,
        tolerance: float = 1e-15,
    ) -> dict:
        """Optimize untill convergence for a single h_subset.
        Args:
            - X: train features
            - y: train targets
            - seed: control random initialisation
            - initial_weights: a vector of initial weights, must have shape (n_features + 1, ) where the element at index 0 is the bias
            - epochs: number of iterations over the training set
            - tf_model: already initialised tensorflow model (initial_weights will be ignored if not None)
            - history: existing history, train loss will be appended to this list
        """
        if tf_model is None:
            np.random.seed(seed)
            tf.random.set_seed(seed)
            tf_model = PickableSequential([Dense(units=1)])
            # intitialize weights
            _ = tf_model(X)

            if initial_weights is not None:
                self.logger.warning(
                    f"Setting initial weights to {initial_weights} instead of randomly initializing"
                )
                tf_model.layers[0].set_weights(
                    [initial_weights[1:].reshape(-1, 1), initial_weights[[0]]]
                )
            elif self.use_fast_lts_initialisation:
                weights, _ = SoftLTS._get_initial_weights_and_augment_input(X, y, seed)
                tf_model.layers[0].set_weights([weights[1:].reshape(-1, 1), weights[[0]]])

        h = int(self.alpha * X.shape[0])
        history = self._train(
            epochs, X, y, tf_model=tf_model, history=history, patience=patience, tolerance=tolerance
        )
        weights = tf_model.layers[0].kernel.numpy().flatten()
        bias = tf_model.layers[0].bias.numpy()

        model = LinearRegression()
        model.intercept_ = bias
        model.coef_ = weights
        best_h_subset = np.argsort(np.abs(y.flatten() - model.predict(X)))[:h]
        train_loss = FastLTS._get_loss_value(X, y, best_h_subset, model)

        return {
            "tf_model": tf_model,
            "lr_model": model,
            "train_loss": train_loss,
            "best_h_subset": best_h_subset,
            "history": history,
        }

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        apply_c_steps: bool = False,
        initial_weights: Optional[np.ndarray] = None,
        patience: Optional[int] = None,
        tolerance: float = 1e-15,
        verbosity: int = logging.INFO,
    ):
        """Fit a softLTS using the SGD.

        Args:
            X : Training features
            y : Training labels
            apply_c_steps (optional): whether to apply c-steps till convergence after the BFGS algorithm (cfr FastLTS). Defaults to False.
            initial_weights (optional): Optionally specify a predefined set of initial weights, must have shape (n_features + 1, ) (including intercept at index 0). Defaults to None.
            verbosity (optional): logging level. Defaults to logging.INFO.
        """
        self.logger.setLevel(verbosity)
        self.logger.info(f"Training models on {self.n_initial_subsets} initial subsets")
        self.results = []
        # train models on initial subsets
        for seed in tqdm(
            range(
                self.random_state * 10,
                self.random_state * 10 + self.n_initial_subsets * self.random_state,
                self.random_state,
            ),
            disable=verbosity > logging.INFO,
        ):
            result = self._single_optimisation(
                X, y, seed=seed, initial_weights=initial_weights, epochs=self.n_initial_iters
            )
            self.results.append(result)

        self.logger.info(f"Finished optimization for {self.n_initial_subsets} initial subsets.")
        best_results = sorted(self.results, key=lambda x: x["train_loss"])[: self.n_best_models]
        self.final_models = []
        for i, result in enumerate(best_results, start=1):
            self.logger.debug(
                f"Optimising until convergence for model {i} out of {self.n_best_models}"
            )
            self.final_models.append(
                self._single_optimisation(
                    X,
                    y,
                    seed=i,
                    tf_model=result["tf_model"],
                    epochs=epochs,
                    history=result["history"],
                    patience=patience,
                    tolerance=tolerance,
                )
            )
        best_result = min(self.final_models, key=lambda x: x["train_loss"])
        self.tf_model = best_result["tf_model"]
        self.model = best_result["lr_model"]
        self.best_h_subset = best_result["best_h_subset"]
        self.train_loss = best_result["train_loss"]
        self.history = best_result["history"]

        if apply_c_steps:
            self.logger.info("Applying c-steps till convergence")
            (
                self.model,
                self.best_h_subset,
                self.train_loss,
                self.n_c_steps,
            ) = FastLTS._apply_C_steps_untill_convergence(
                self.model,
                self.train_loss,
                X,
                y,
                int(self.alpha * X.shape[0]),
                tolerance=tolerance,
                logger=self.logger,
            )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class SoftLMS(SoftLTS):
    """
    Implementation of Least Median Squares using the soft sorting operator as defined in the paper
    `Fast Differentiable Sorting and Ranking Mathieu Blondel, Olivier Teboul, Quentin Berthet, Josip Djolonga In proceedings of ICML 2020 arXiv:2002.08871`
    using the python implementation from `https://github.com/google-research/fast-soft-sort`
    """

    def __init__(
        self,
        alpha: float = 0.5,
        regularization: str = "l2",
        regularization_strength: float = 1.0,
        random_state: int = 42,
        n_initial_subsets: int = 1,
        n_initial_iters: int = 1,
        n_best_models: int = 1,
    ):
        assert n_initial_subsets >= n_best_models, (
            f"`n_initial_subsets` must be greater than or equal to `n_best_models`, "
            f"but were {n_initial_subsets} and {n_best_models} respectively"
        )

        self.alpha = alpha
        self.regularization = regularization  # options: ('l2', 'kl') (quadratic or entropic)
        self.regularization_strength = regularization_strength  # epsilon
        self.random_state = random_state
        self.n_initial_subsets = n_initial_subsets
        self.n_initial_iters = n_initial_iters
        self.n_best_models = n_best_models
        self.logger = logging.getLogger("soft_lts")

    def _single_optimisation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seed: int = 42,
        initial_weights: Optional[np.ndarray] = None,
        calc_gradients: bool = True,
        max_iter: int = 300,
        ftol: float = 1e-15,
        verbosity: int = logging.INFO,
        convergence_warning: bool = False,
        normalize_gradients: bool = False,
    ) -> dict:
        """Optimisation until convergence or max_iter using the BFGS algorithm"""
        result = super()._single_optimisation(
            X=X,
            y=y,
            seed=seed,
            initial_weights=initial_weights,
            calc_gradients=calc_gradients,
            max_iter=max_iter,
            ftol=ftol,
            verbosity=verbosity,
            convergence_warning=convergence_warning,
            normalize_gradients=normalize_gradients,
        )
        # overwrite train_loss
        h_idx = result["best_h_subset"][-1]
        result["train_loss"] = FastLTS._get_loss_value(X, y, [h_idx], result["lr_model"])

        return result

    @staticmethod
    def _compute_soft_loss_and_gradient(
        weights: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        h: int,
        eps: float = 1.0,
        regul: str = "l2",
        return_gradients: bool = True,
        normalize_gradients: bool = False,
        logger: logging.Logger = None,
    ) -> Union[float, Tuple[float, np.ndarray]]:
        """Given linear regression coefficients, input data and soft sort parameters,
        calculate the sum of the `h` minumum soft squared residuals as well as the gradient with respect to the weights

        Args:
            - weights (ndarray): 1d array of shape (n_features + 1,), the first element is expected to be the intercept
            - X (ndarray): 2d array of shape (n_obs, n_features + 1) , the first element is expected to contain ones
            - y (ndarray): 1d array of shape (n_obs, )
            - h (int): take the h-th sample for calculating the loss and gradients (alpha x n_obs)
            - eps (float): regularization strength for soft sort. Default = 1.0
            - regul (strt): type of regularization for soft sort, either `l2` or `kl`. Default = 'l2'
        """
        if not logger:
            logger = logging.getLogger("soft_lts")
            logger.setLevel(logging.INFO)
        # start = time.time()
        W = weights.reshape((-1, 1))

        y_pred = np.matmul(X, W)  # shape = (n_input, 1)
        residuals = y_pred - y.reshape(-1, 1)  # shape = (n_input, 1)
        squared_residuals = residuals**2  # shape = (n_input, 1)

        # soft_sort residuals
        softsort = numpy_ops.SoftSort(
            values=squared_residuals.flatten(),
            direction="ASCENDING",
            regularization_strength=eps,
            regularization=regul,
        )

        soft_squared_residuals = softsort.compute()  # shape = (n_input, )
        # note that we are not taking the actual median in case there is an even number of observations
        # this would require taking the mean of observations h-1 and h
        # LTS: soft_loss = np.sum(soft_squared_residuals[:h]) / h
        soft_loss = soft_squared_residuals[(h - 1)]
        # print(
        #     soft_loss,
        #     np.sum(soft_squared_residuals[:h]) / h,
        #     np.sum(soft_squared_residuals[(h - 3) : h]) / 3,
        # )
        if not return_gradients:
            return soft_loss

        gradient_of_loss_vector_to_weights = 2 * residuals * X  # shape = (n_input x n_feat + 1)

        # LTS
        # soft_gradients = (
        #   jacobian_matrix_mult(softsort, gradient_of_loss_vector_to_weights, max_idx=h)[
        #         :h, :
        #   ].sum(axis=0)
        soft_gradients = jacobian_matrix_mult(
            softsort, gradient_of_loss_vector_to_weights, max_idx=h
        )[(h - 1), :]
        # shape = (n_features + 1, )

        if normalize_gradients:
            soft_gradients /= np.linalg.norm(soft_gradients)

        return soft_loss, soft_gradients


class SoftLTS_MLP(RegressorMixin, BaseEstimator):
    def __init__(
        self,
        alpha: float = 0.5,
        regularization: str = "l2",
        regularization_strength: float = 1.0,
        learning_rate: float = 0.1,
        n_initial_subsets: int = 1,
        n_initial_iters: int = 1,
        n_best_models: int = 1,
        random_state: int = 42,
        layers: list[Layer] = [Dense(units=1, activation="linear")],
    ):
        self.alpha = alpha
        self.regularization = regularization  # options: ('l2', 'kl') (quadratic or entropic)
        self.regularization_strength = regularization_strength  # epsilon
        self.random_state = random_state
        self.layers = layers
        self.tf_model = PickableSequential(layers)
        self.learning_rate = learning_rate
        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        self.n_initial_subsets = n_initial_subsets
        self.n_initial_iters = n_initial_iters
        self.n_best_models = n_best_models
        self.logger = logging.getLogger("SoftLTS_MLP")

    def _train_step(self, X: np.ndarray, y: np.ndarray, tf_model: PickableSequential) -> float:
        with tf.GradientTape() as tape:
            y_pred = tf_model(X, training=True)
            loss_value = lts_loss_function(
                y, y_pred, h=int(len(X) * self.alpha), epsilon=self.regularization_strength
            )

        grads = tape.gradient(loss_value, tf_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, tf_model.trainable_variables))

        return loss_value.numpy().mean()

    def _train(
        self,
        epochs: int,
        X: np.ndarray,
        y: np.ndarray,
        tf_model: Optional[PickableSequential] = None,
        history: Optional[List[float]] = None,
        patience: Optional[int] = None,
    ) -> List[float]:
        patience = patience if not patience is None else epochs
        if tf_model is None:
            tf_model = self.tf_model
        history = [] if history is None else history
        counter = 0
        best_loss = np.inf
        pbar = tqdm(range(epochs), disable=self.logger.level > logging.DEBUG)
        # TODO: add storing best model weights
        for i in pbar:
            loss = self._train_step(X, y, tf_model)
            history.append(loss)
            pbar.set_postfix({"Train loss": loss})
            if loss < best_loss:
                best_loss = loss
                counter = 0
            else:
                counter += 1
            if counter > patience:
                self.logger.warning(
                    f"Stopping early after {i} iterations. "
                    f"No loss improvements for the last {counter} iterations."
                )
                break

        return history

    def _single_optimisation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seed: int,
        initial_weights: Optional[list[np.ndarray]] = None,
        epochs: int = 100,
        tf_model: Optional[PickableSequential] = None,
        history: Optional[List[float]] = None,
        patience: Optional[int] = None,
    ) -> dict:
        if tf_model is None:
            np.random.seed(seed)
            tf.random.set_seed(seed)
            tf_model = PickableSequential(self.layers)
            # intitialize weights
            _ = tf_model(X)

            if initial_weights is not None:
                self.logger.warning(f"Setting initial weights instead of randomly initializing")
                tf_model.set_weights(initial_weights)

        h = int(self.alpha * X.shape[0])
        history = self._train(epochs, X, y, tf_model=tf_model, history=history, patience=patience)

        best_h_subset = np.argsort(np.abs(y.flatten() - tf_model.predict(X).flatten()))[:h]
        train_loss = FastLTS._get_loss_value(X, y, best_h_subset, tf_model)

        return {
            "model": tf_model,
            "train_loss": train_loss,
            "best_h_subset": best_h_subset,
            "history": history,
        }

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        patience: Optional[int] = None,
        initial_weights: Optional[list[np.ndarray]] = None,
        verbosity: int = logging.INFO,
    ):
        """Fit a MLP with Soft LTS objective.

        Args:
            X (np.ndarray): Training features
            y (np.ndarray): Training labels
            epochs: number of loops over the training data
            patience: how many iterations to wait before stopping early, if None, no early stopping is applied
            initial_weights (optional): Optionally specify a predefined set of initial weights (including intercept at index 0). Defaults to None.
            verbosity (int, optional): logging level. Defaults to logging.INFO.
        """
        self.logger.setLevel(verbosity)
        self.logger.info(f"Training models on {self.n_initial_subsets} initial subsets")
        self.results = []
        # train models on initial subsets
        for seed in tqdm(
            range(
                self.random_state * 10,
                self.random_state * 10 + self.n_initial_subsets * self.random_state,
                self.random_state,
            ),
            disable=verbosity > logging.DEBUG,
        ):
            result = self._single_optimisation(
                X, y, seed=seed, initial_weights=initial_weights, epochs=self.n_initial_iters
            )
            self.results.append(result)

        self.logger.info(f"Finished optimization for {self.n_initial_subsets} initial subsets.")
        best_results = sorted(self.results, key=lambda x: x["train_loss"])[: self.n_best_models]
        self.final_models = []
        for i, result in enumerate(best_results, start=1):
            self.logger.debug(
                f"Optimising until convergence for model {i} out of {self.n_best_models}"
            )
            self.final_models.append(
                self._single_optimisation(
                    X,
                    y,
                    seed=i,
                    tf_model=result["model"],
                    epochs=epochs,
                    history=result["history"],
                    patience=patience,
                )
            )
        best_result = min(self.final_models, key=lambda x: x["train_loss"])
        self.model = best_result["model"]
        self.best_h_subset = best_result["best_h_subset"]
        self.train_loss = best_result["train_loss"]
        self.history = best_result["history"]

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
