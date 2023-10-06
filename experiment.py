"""
This modules contains the logic to replicate the LTS experiment from the paper
`Fast Differentiable Sorting and Ranking Mathieu Blondel, Olivier Teboul, Quentin Berthet, Josip Djolonga In proceedings of ICML 2020 arXiv:2002.08871`
"""
from __future__ import annotations
import itertools
import logging
import time
import copy
import datetime
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple, HandlerLine2D

from typing import Any, Dict, Literal, Tuple, Optional, List, Iterable, Callable, Union
from enum import Enum, auto
from dataclasses import dataclass, field
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import tensorflow as tf

from soft_lts import ClassicLMS, FastLTS, SoftLMS, SoftLTS, SoftLTSSGD
from data_loader import DataLoader
from custom_wrapping import XYWrapRegressor, XWrapRegressor
from data_simulator import DataSimulatorA09, DataSimulatorSimple
from utils import get_smart_initialisation_weights, get_smart_initialisation_weights_alt
from mm_estimator import get_mm_estimator_coefficients
from rpy2.rinterface_lib.embedded import RRuntimeError

MODEL_DICT = {
    "FAST_LTS": FastLTS,
    "SOFT_LTS": SoftLTS,
    "SOFT_LTS_SGD": SoftLTSSGD,
    "OLS": LinearRegression,
    "SOFT_LMS": SoftLMS,
    "LMS": ClassicLMS,
    "XY_WRAP": XYWrapRegressor,
    "X_WRAP": XWrapRegressor,
}

OPTIMIZER_DICT = {
    "Adam": tf.optimizers.Adam,
    "SGD": tf.optimizers.SGD,
    "RMSprop": tf.optimizers.RMSprop,
}


class Model(Enum):
    FAST_LTS = auto()
    SOFT_LTS = auto()
    SOFT_LTS_SGD = auto()
    OLS = auto()
    SOFT_LMS = auto()
    LMS = auto()
    XY_WRAP = auto()
    X_WRAP = auto()
    MM_ESTIMATOR = auto()
    S_ESTIMATOR = auto()


@dataclass
class ParamGrid:
    name: str
    model: Model
    smart_init: bool = field(default=False)
    smart_init_alt: bool = field(default=False)
    fit_params: Dict[str, Any] = field(default_factory=dict)
    grid_params: Dict[str, Iterable[Any]] = field(default_factory=dict)


class Experiment:
    def __init__(
        self,
        test_perc: float = 0.2,
        repetitions: int = 10,
        nr_cv_folds: int = 5,
        datasets: List[
            Literal[
                "bodyfat",
                "cadata",
                "simulated_a09",
                "simulated_simple",
                "abalone",
                "hardware",
                "ale",
                "power",
            ]
        ] = ["bodyfat", "cadata"],
        datapath: str = "./Data/{}.txt",
        scaler: Optional[Callable[..., TransformerMixin]] = None,
        outlier_perc: Iterable[float] = [0.1, 0.2, 0.3, 0.4, 0.5],
        leverage_perc: float = 0.0,
        a09_simulator_params: Optional[Dict[str, Any]] = None,
        simple_simulator_params: Optional[Dict[str, Any]] = None,
        param_grids: Optional[List[ParamGrid]] = None,
    ):
        self.test_perc = test_perc
        self.repetitions = repetitions
        self.nr_cv_folds = nr_cv_folds
        self.datasets = datasets
        self.datapath = datapath
        self.outlier_perc = outlier_perc
        self.leverage_perc = leverage_perc  # relative to outlier perc
        self.scaler = scaler
        self.a09_simulator_params = a09_simulator_params if a09_simulator_params is not None else {}
        self.simple_simulator_params = simple_simulator_params if simple_simulator_params else {}
        self.param_grids = (
            param_grids
            if param_grids is not None
            else [
                ParamGrid(
                    name="FastLTS",
                    model=Model.FAST_LTS,
                    fit_params={},
                    grid_params={
                        "alpha": [0.5, 0.6, 0.7, 0.8, 0.9, 1],
                        "n_initial_subsets": [500],
                        "n_best_models": [10],
                        "n_initial_c_steps": [2],
                    },
                ),
                ParamGrid(
                    name="SoftLTS",
                    model=Model.SOFT_LTS,
                    fit_params={"apply_c_steps": False, "max_iter": 300},
                    grid_params={
                        "alpha": [0.5, 0.6, 0.7, 0.8, 0.9, 1],
                        "regularization_strength": np.logspace(-3, 4, num=10),
                        "n_initial_subsets": [1],
                        "n_best_models": [1],
                        "n_initial_iters": [1],
                    },
                ),
                ParamGrid(
                    name="SoftLTSSGD",
                    model=Model.SOFT_LTS_SGD,
                    fit_params={"apply_c_steps": False, "epochs": 100},
                    grid_params={
                        "alpha": [0.5, 0.6, 0.7, 0.8, 0.9, 1],
                        "regularization_strength": np.logspace(-3, 4, num=10),
                        "learning_rate": [0.3, 0.1, 0.03, 0.01, 0.003, 0.001],
                        "n_initial_subsets": [1],
                        "n_best_models": [1],
                        "n_initial_iters": [1],
                        "use_fast_lts_initialisation": [False],
                    },
                ),
                ParamGrid(name="OLS", model=Model.OLS),
            ]
        )
        self.results = None
        self.bad_results = []

    @staticmethod
    def _verbose_print(msg: str, verbose: bool):
        if verbose:
            print(msg)

    def run_grid(
        self,
        param_grid: ParamGrid,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        context: dict,
        verbose: bool = True,
        disable_gridsearch: bool = False,
    ) -> dict:
        self._verbose_print(f"\t\t\tTraining `{param_grid.name}`", verbose)
        results = dict(model=param_grid.name, **context)
        start = time.time()

        if param_grid.model in [Model.MM_ESTIMATOR, Model.S_ESTIMATOR]:
            try:
                coef = get_mm_estimator_coefficients(X_train, y_train)[param_grid.model.name]
                if coef is None:
                    self._verbose_print(f"\t\t\tNo coefficients, returning negative R2", verbose)
                    r2 = -100
                else:
                    r2 = r2_score(
                        y_test,
                        np.hstack((np.ones((X_test.shape[0], 1)), X_test)) @ coef,
                    )
                results.update(dict(test_rsquared=r2))
            except (AttributeError, RRuntimeError) as e:
                self._verbose_print(f"\t\t\tError {e}, returning negative R2", verbose)
                results.update(dict(test_rsquared=-100))
            end = time.time()
            results.update(dict(total_duration=(end - start)))
            return results

        estimator = (
            MODEL_DICT[param_grid.model.name](random_state=(context["repetition"] + 1) * 10)
            if param_grid.model.name not in [Model.OLS.name, Model.XY_WRAP.name, Model.X_WRAP.name]
            else MODEL_DICT[param_grid.model.name]()
        )

        if param_grid.grid_params:
            if not disable_gridsearch:
                gridsearch = GridSearchCV(
                    estimator, param_grid.grid_params, n_jobs=-1, cv=self.nr_cv_folds
                )

                gridsearch.fit(X_train, y_train, verbosity=logging.ERROR, **param_grid.fit_params)
                results.update(
                    dict(
                        mean_fit_time=gridsearch.cv_results_["mean_fit_time"][
                            gridsearch.best_index_
                        ],
                        test_rsquared=gridsearch.score(X_test, y_test),
                        **gridsearch.best_params_,
                    )
                )
                estimator = gridsearch.best_estimator_

                if results["test_rsquared"] < 0.5:
                    self.bad_results.append(
                        {
                            "X_train": X_train,
                            "y_train": y_train,
                            "X_test": X_test,
                            "y_test": y_test,
                            "rsquared": results["test_rsquared"],
                            "gridsearch": gridsearch,
                            "estimator": estimator,
                            "context": context,
                        }
                    )
            else:
                # print("WARNING: not running gridsearch")
                estimator.set_params(
                    **{key: value[0] for key, value in param_grid.grid_params.items()}
                )
                estimator.fit(X_train, y_train, verbosity=logging.ERROR, **param_grid.fit_params)
                try:
                    results.update({"test_rsquared": r2_score(y_test, estimator.predict(X_test))})
                except ValueError as ve:
                    print(f"ERROR: {ve}")
                    results.update({"test_rsquared": np.inf})

            self._update_results(
                results, estimator, X_train, y_train, X_test, y_test, param_grid.model
            )

        else:
            estimator.fit(X_train, y_train, **param_grid.fit_params)
            results.update(dict(test_rsquared=estimator.score(X_test, y_test)))
        end = time.time()
        results.update(dict(total_duration=(end - start)))

        return results

    def _update_results(
        self,
        results: dict,
        estimator: BaseEstimator,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model: Model,
    ) -> None:
        # append train and test trimmed loss
        h_train = int(estimator.get_params().get("alpha", 0.5) * len(X_train))
        h_test = int(estimator.get_params().get("alpha", 0.5) * len(X_test))
        train_idx_sorted = np.argsort(
            np.abs(y_train.flatten() - estimator.predict(X_train).flatten())
        )
        test_idx_sorted = np.argsort(np.abs(y_test.flatten() - estimator.predict(X_test).flatten()))

        results.update(
            {
                "train_trimmed_loss": FastLTS._get_loss_value(
                    X_train,
                    y_train,
                    train_idx_sorted[:h_train],
                    estimator,
                ),
                "train_lms_loss": ClassicLMS._get_loss_value(
                    X_train,
                    y_train,
                    estimator,
                ),
                "test_trimmed_loss": FastLTS._get_loss_value(
                    X_test,
                    y_test,
                    test_idx_sorted[:h_test],
                    estimator,
                ),
                "test_lms_loss": ClassicLMS._get_loss_value(
                    X_train,
                    y_train,
                    estimator,
                ),
            }
        )

        if model == Model.SOFT_LTS:
            results.update(
                dict(
                    converged=estimator.bfgs_result.success,
                    bfgs_iterations=estimator.bfgs_result.nit,
                )
            )
        elif model == Model.SOFT_LTS_SGD:
            results.update(dict(history=estimator.history))

    def run(
        self,
        verbose: bool = True,
        store_intermediate: bool = True,
        path: str = None,
        disable_gridsearch: bool = False,
    ) -> None:
        """Run simulations

        Args:
            verbose (optional): whether or not to print progress. Defaults to True.
            store_intermediate (optional): whether to store intermediate results after every repetition. Defaults to True.
            path (optional): where to store the results. Defaults to None.
        """
        if self.results is None or (isinstance(self.results, pd.DataFrame) and self.results.empty):
            self.results = pd.DataFrame()
            start_iter = 0
        else:
            start_iter = int(self.results["repetition"].max()) + 1
            print(f"Resuming from iteration {start_iter} out of {self.repetitions}")

        simulator_a09 = DataSimulatorA09(**self.a09_simulator_params)
        simulator_simple = DataSimulatorSimple(**self.simple_simulator_params)
        for repetition in range(start_iter, self.repetitions):
            self._verbose_print(f"Round {repetition}", verbose)
            for dataset in self.datasets:
                self._verbose_print(f"\tDataset: {dataset}", verbose)
                for outlier_perc in self.outlier_perc:
                    self._verbose_print(f"\t\tOutlier perc: {outlier_perc}", verbose)
                    if dataset == "simulated_a09":
                        X_train, X_test, y_train, y_test = simulator_a09.get_data(
                            test_perc=self.test_perc,
                            outlier_perc=outlier_perc,
                            random_seed=repetition,
                        )
                    elif dataset == "simulated_simple":
                        X_train, X_test, y_train, y_test = simulator_simple.get_data(
                            test_perc=self.test_perc,
                            outlier_perc=outlier_perc,
                            random_seed=repetition,
                        )
                    else:
                        X_train, X_test, y_train, y_test = DataLoader(
                            datapath=self.datapath.format(dataset),
                            test_perc=self.test_perc,
                            outlier_perc=outlier_perc,
                            bad_leverage_perc=self.leverage_perc,
                            random_state=repetition,
                            scaler=self.scaler,
                        ).get_data()

                    if any(pg.smart_init for pg in self.param_grids):
                        self._verbose_print(
                            "\t\t\tCalculating smart initialisation weights", verbose
                        )
                        initial_weights = get_smart_initialisation_weights(
                            X_train, y_train, 0.5, rescale=True
                        )
                        for pg in self.param_grids:
                            if pg.smart_init:
                                pg.fit_params.update({"initial_weights": initial_weights})
                    if any(pg.smart_init_alt for pg in self.param_grids):
                        self._verbose_print(
                            "\t\t\tCalculating smart initialisation weights (alt)",
                            verbose,
                        )
                        initial_weights = get_smart_initialisation_weights_alt(
                            X_train, y_train, 0.5, rescale=True
                        )
                        for pg in self.param_grids:
                            if pg.smart_init_alt:
                                pg.fit_params.update({"initial_weights": initial_weights})

                    context = {
                        "outlier_perc": outlier_perc,
                        "dataset": dataset,
                        "repetition": repetition,
                    }
                    for param_grid in self.param_grids:
                        self.results = pd.concat(
                            [
                                self.results,
                                pd.DataFrame(
                                    [
                                        self.run_grid(
                                            param_grid=param_grid,
                                            X_train=X_train,
                                            y_train=y_train,
                                            X_test=X_test,
                                            y_test=y_test,
                                            context=context,
                                            verbose=verbose,
                                            disable_gridsearch=disable_gridsearch,
                                        )
                                    ]
                                ),
                            ],
                            axis=0,
                            ignore_index=True,
                        )
            if store_intermediate:
                if path is None:
                    path = f'./Output/soft_lts_experiment_{datetime.datetime.now().strftime("%Y%m%d")}.pkl'
                print(f"Storing intermediate results to {path}")
                self.save(path)

        self._verbose_print("<<<<DONE>>>>", verbose)

    def save(self, path: str):
        if self.results is None:
            raise NotFittedError("Eperiment has not run yet.")
        print(f"Storing results to {path}")
        param_grids = copy.deepcopy(self.param_grids)
        for param_grid in param_grids:
            if "optimizer" in param_grid.grid_params:
                param_grid.grid_params.update(
                    {"optimizer": [o.get_config() for o in param_grid.grid_params["optimizer"]]}
                )

        joblib.dump(
            {
                "test_perc": self.test_perc,
                "repetitions": self.repetitions,
                "nr_cv_folds": self.nr_cv_folds,
                "datasets": self.datasets,
                "datapath": self.datapath,
                "outlier_perc": self.outlier_perc,
                "leverage_perc": self.leverage_perc,
                "scaler": self.scaler,
                "results": self.results,
                "a09_simulator_params": self.a09_simulator_params,
                "simple_simulator_params": self.simple_simulator_params,
                "param_grids": param_grids,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> Experiment:
        data = joblib.load(path)
        experiment = cls(**{k: v for k, v in data.items() if k != "results"})
        experiment.results = data["results"]
        for param_grid in experiment.param_grids:
            if "optimizer" in param_grid.grid_params:
                param_grid.grid_params["optimizer"] = [
                    OPTIMIZER_DICT[o["name"]].from_config(o)
                    for o in param_grid.grid_params["optimizer"]
                ]

        return experiment

    def plot(
        self,
        ylim: Union[
            Tuple[Optional[float], Optional[float]],
            str,
            dict[str, Tuple[Optional[float], Optional[float]]],
        ] = (None, None),
        metric: str = "test_rsquared",
        agg: str = "mean",
        fill_between: bool = False,
        models: Union[str, List[str], Dict[str, str]] = "all",
        datasets: Union[str, List[str]] = "all",
        axs: Optional[Iterable[Axes]] = None,
        layout: Tuple[int] = None,
        figsize: Tuple[int] = (20, 6),
        title_dict: Optional[dict] = None,
        linewidth: int = 2,
        ylabel: Optional[str] = None,
        append_agg_to_legend: bool = True,
        bbox_to_anchor: Tuple[float, float] = (0.5, 1.05),
        n_legend_col: int = 3,
        x_label: Optional[str] = None,
    ):
        if title_dict is None:
            title_dict = {}
        if axs is None:
            layout = (1, len(self.datasets)) if layout is None else layout
            fig, axs = plt.subplots(*layout, figsize=figsize, squeeze=False)

        models_to_plot = list(self.results["model"].unique()) if models == "all" else list(models)
        datasets = self.datasets if datasets == "all" else datasets
        line_styles = ["solid", "dashed", "dashdot", "dotted"]
        for i, (ax, dataset) in enumerate(zip(np.array(axs).flatten(), datasets)):
            plot_df = (
                self.results[self.results["dataset"] == dataset]
                .groupby(["model", "outlier_perc"])[metric]
                .agg(mean="mean", median="median", min="min", max="max")
            )
            for model, ls in zip(models_to_plot, itertools.cycle(line_styles)):
                label = models.get(model, model) if isinstance(models, dict) else model
                if append_agg_to_legend:
                    label += f"_{agg}"
                plot_df.loc[model].reset_index().plot(
                    x="outlier_perc", y=agg, ax=ax, label=label, lw=linewidth, ls=ls
                )
                if fill_between:
                    ax.fill_between(
                        x=plot_df.loc[model].reset_index()["outlier_perc"],
                        y1=plot_df.loc[model].reset_index()["min"],
                        y2=plot_df.loc[model].reset_index()["max"],
                        alpha=0.4,
                        label=model + "_min/max",
                    )
            ax.set_title(title_dict.get(dataset, dataset))
            if ylim == "adaptive":
                ylim_ax = (
                    max(plot_df[agg].quantile(0.2) - 0.1, -0.1),
                    min(plot_df[agg].max() + 0.05, 1),
                )
            elif isinstance(ylim, dict):
                ylim_ax = ylim.get(dataset)
            else:
                ylim_ax = ylim
            ax.set_ylim(ylim_ax)
            ax.get_legend().remove()
            ax.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5])
            if x_label is not None:
                ax.set_xlabel(x_label)
        if len(axs.shape) == 2:
            for ax in axs[:, 0]:
                ax.set_ylabel(metric if ylabel is None else ylabel)
        else:
            np.array(axs).flatten()[0].set_ylabel(metric)
        handles, labels = np.array(axs).flatten()[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=n_legend_col,
            bbox_to_anchor=bbox_to_anchor,
        )
        fig.tight_layout()
        return axs

    def plot_fit_times(
        self,
        attribute: str = "mean_fit_time",
        models: Union[str, List[str], Dict[str, str]] = "all",
        datasets: Union[str, List[str]] = "all",
        agg: str = "mean",
        by_outlier_perc: bool = False,
        ylim: Optional[Tuple[Optional[float], Optional[float]]] = None,
        bbox_to_anchor: Tuple[float, float] = (0.5, 1.3),
        n_legend_col: int = 3,
        y_log_scale: bool = True,
        dataset_dict: Optional[dict] = None,
        x_ticklabel_rot: int = 90,
        figsize: tuple[int, int] = (25, 6),
    ):
        if models != "all":
            results = self.results.loc[self.results["model"].isin(list(models))]
            models_to_plot = list(models)
        else:
            results = self.results
            models_to_plot = [p.name for p in self.param_grids]
        datasets = self.datasets if datasets == "all" else datasets
        results = results.loc[results["dataset"].isin(datasets)]
        if by_outlier_perc:
            outlier_perc = sorted(results["outlier_perc"].unique())
            fig, axs = plt.subplots(1, len(outlier_perc), figsize=figsize)
            max_value = (
                results.groupby(["dataset", "model", "outlier_perc"])[attribute].agg(agg).max()
            )
            for outlier_p, ax in zip(outlier_perc, axs.flatten()):
                plot_data = (
                    results.loc[results["outlier_perc"] == outlier_p]
                    .groupby(["dataset", "model"])[attribute]
                    .agg(agg)
                    .unstack()[models_to_plot]
                )
                plot_data = plot_data.reindex(models_to_plot, axis=1)
                plot_data.plot.bar(ylabel=attribute.replace("_", " ").capitalize(), ax=ax)
                ax.set_title(outlier_p)
                ax.get_legend().remove()
                if ylim is None:
                    ax.set_ylim((0, max_value * 1.1))
                else:
                    ax.set_ylim(ylim)
                ax.set_xticklabels(ax.get_xticks(), rotation=x_ticklabel_rot)

            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center")
            return axs
        else:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            plot_data = (
                results.groupby(["dataset", "model"])[attribute].agg(agg).unstack()[models_to_plot]
            )
            plot_data = plot_data.reindex(models_to_plot, axis=1)
            if isinstance(models, dict):
                plot_data.rename(columns=lambda c: models.get(c, c), inplace=True)
            if dataset_dict is not None:
                plot_data.index = plot_data.index.map(dataset_dict)
            ylabel = attribute.replace("_", " ").capitalize() + " (seconds)"
            ax = plot_data.plot.bar(ylabel=ylabel, ax=ax, logy=y_log_scale)
            ax.legend(
                loc="upper center",
                bbox_to_anchor=bbox_to_anchor,
                ncol=n_legend_col,
                fancybox=True,
                shadow=True,
            )
            ax.grid(axis="y", linestyle="dotted", alpha=0.8)
            if ylim is not None:
                ax.set_ylim(ylim)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=x_ticklabel_rot)
            return ax

    def plot_r2_and_fit_time(
        self,
        figsize: tuple[int, int] = (20, 8),
        dataset_dict: Optional[dict[str, str]] = None,
        datasets: Optional[list[str]] = None,
        models: Optional[Iterable[str]] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        ncol: int = 2,
        n_legend_col: int = 4,
        linewidth: int = 4,
        legend_bbox_to_anchor: tuple[float, float] = (0.5, 1.15),
    ):
        dataset_dict = {} if dataset_dict is None else dataset_dict
        datasets = self.datasets if datasets is None else datasets
        models = list(self.results["model"].unique()) if models is None else list(models)
        fig = plt.figure(figsize=figsize)
        axd = fig.subplot_mosaic(
            np.array(np.array_split(datasets, ncol)).T.tolist() + [["duration", "duration"]]
        )
        line_styles = ["solid", "dashed", "dashdot", "dotted"]

        for i, dataset in enumerate(datasets):
            ax = axd[dataset]
            plot_df = (
                self.results[self.results["dataset"] == dataset]
                .groupby(["model", "outlier_perc"])["test_rsquared"]
                .agg(mean="mean", median="median", min="min", max="max")
            )
            for model, ls in zip(models, itertools.cycle(line_styles)):
                label = models.get(model, model) if isinstance(models, dict) else model
                plot_df.loc[model].reset_index().plot(
                    x="outlier_perc",
                    y="mean",
                    ax=ax,
                    label=label,
                    lw=linewidth,
                    ls=ls,
                    marker="o",
                    ms=10,
                    alpha=0.7,
                )
            ax.set_title(dataset_dict.get(dataset, dataset), fontsize=25)
            ax.set_ylim(
                (
                    max(plot_df["mean"].quantile(0.2) - 0.1, -0.1),
                    min(plot_df["mean"].max() + 0.05, 1),
                )
            )
            ax.get_legend().remove()
            xticks = [0.1, 0.2, 0.3, 0.4, 0.5]
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticks, fontsize=12)
            if x_label is not None:
                ax.set_xlabel(x_label, fontsize=20)
            if i <= len(datasets) / ncol - 1:
                ax.set_ylabel("test_rsquared" if y_label is None else y_label, fontsize=20)
            ax.tick_params(axis="both", labelsize=20)
            ax.grid(axis="y", linestyle="dotted", alpha=0.8)

        plot_data = (
            self.results.groupby(["dataset", "model"])["total_duration"]
            .agg("mean")
            .unstack()[models]
        )
        plot_data = plot_data.reindex(models, axis=1).loc[datasets]
        if isinstance(models, dict):
            plot_data.rename(columns=lambda c: models.get(c, c), inplace=True)
        plot_data.index = plot_data.index.map(dataset_dict)
        ax = plot_data.plot.bar(ax=axd["duration"], logy=True, legend=False)
        ax.grid(axis="y", linestyle="dotted", alpha=0.8)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.set_ylabel("Total fit duration (seconds)", fontsize=20)
        ax.set_xlabel("Dataset", fontsize=20)
        ax.tick_params(axis="both", labelsize=20)
        duration_handles, duration_labels = ax.get_legend_handles_labels()

        r2_handles, r2_labels = [a for n, a in axd.items() if n != " duration"][
            0
        ].get_legend_handles_labels()
        r2_handles = [copy.copy(h) for h in r2_handles]
        for h in r2_handles:
            h.set_marker("")
        fig.legend(
            [(h1, h2) for h1, h2 in zip(r2_handles, duration_handles)],
            duration_labels,
            loc="upper center",
            ncol=n_legend_col,
            bbox_to_anchor=legend_bbox_to_anchor,
            handler_map={
                tuple: HandlerTuple(ndivide=None, pad=1),
                Line2D: HandlerLine2D(numpoints=2, marker_pad=1.5),
            },
            fontsize=20,
            borderpad=1,
            handlelength=3,
            markerscale=2,
        )
        # for h in r2_handles:
        #     h.set_marker('o')
        #     h.set_markersize(3)
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.1, hspace=0.3)
        return fig, axd
