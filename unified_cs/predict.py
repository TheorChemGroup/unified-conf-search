import numpy as np

import gpflow

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.ops.numpy_ops import np_config

import trieste
from trieste.data import Dataset
from trieste.space import Box
from trieste.models.gpflow.models import GaussianProcessRegression
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.acquisition.function import ExpectedImprovement
from trieste.ask_tell_optimization import AskTellOptimizerNoTraining

from .improvement_variance import ImprovementVariance
from .pi_kernel import (
    FullPotentialFunction,
    TransformKernel,
    PotentialCoefs,
)
from .config import BOConfig
from pydantic.dataclasses import dataclass
from pydantic import ConfigDict

import logging
from typing import Optional

np_config.enable_numpy_behavior()

VERY_HIGH_ENERGY = 100.0


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class Dof:
    lowest_value: float
    highest_value: float
    atoms: list[int]  # Indexing from 0


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ConformationInfo:
    xyzs: Optional[np.ndarray]
    dofs: np.ndarray
    energy: Optional[float]
    minimum: bool
    corrupted: bool = False
    source: str = "Unknown"


class BOPredictor:

    def __init__(self, dofs: list[Dof], config: BOConfig):
        self.config = config
        self.dofs = dofs
        self.search_space = trieste.space.Box(
            lower=[dof.lowest_value for dof in dofs],
            upper=[dof.highest_value for dof in dofs])

    def predict_new_dofs(
        self,
        conformations: list[ConformationInfo],
        coeffs: PotentialCoefs,
    ) -> np.ndarray:
        # assert all(not x.corrupted for x in conformations)
        dataset = self._build_dataset_for_confs(conformations)
        logging.debug(f"Dataset is {dataset}")
        predicted_dofs = self._predict_for_dataset(dataset, coeffs)
        logging.debug(f"Predicted DOFs are {predicted_dofs}")
        return predicted_dofs

    def _build_dataset_for_confs(
        self,
        conformations: list[ConformationInfo],
    ) -> Dataset:
        logging.info("Start generating dataset")
        lowest_energy = min(conf.energy for conf in conformations
                            if not conf.corrupted)

        dataset = Dataset(
            tf.constant([conf.dofs for conf in conformations], dtype="double"),
            tf.constant(
                [(conf.energy -
                  lowest_energy) if not conf.corrupted else VERY_HIGH_ENERGY
                 for conf in conformations],
                dtype="double").reshape(len(conformations), 1),
        )
        logging.info("Done generating dataset")
        return dataset

    def _predict_for_dataset(
        self,
        dataset: Dataset,
        coeffs: PotentialCoefs,
    ) -> np.ndarray:
        # 1
        kernel = gpflow.kernels.White(0.001) + gpflow.kernels.Periodic(
            gpflow.kernels.RBF(variance=0.07,
                               lengthscales=0.005,
                               active_dims=[i for i in range(len(self.dofs))]),
            period=[dof.highest_value - dof.lowest_value for dof in self.dofs])
        kernel.kernels[1].base_kernel.lengthscales.prior = (
            tfp.distributions.LogNormal(
                loc=tf.constant(0.005, dtype=tf.float64),
                scale=tf.constant(0.001, dtype=tf.float64),
            ))

        potential_func = FullPotentialFunction(coeffs)
        kernel += TransformKernel(
            potential_func,
            gpflow.kernels.RBF(variance=0.12,
                               lengthscales=0.005,
                               active_dims=[i for i in range(len(self.dofs))]))
        kernel.kernels[2].base_kernel.lengthscales.prior = (
            tfp.distributions.LogNormal(
                loc=tf.constant(0.005, dtype=tf.float64),
                scale=tf.constant(0.001, dtype=tf.float64),
            ))

        # 2
        gpr = gpflow.models.GPR(dataset.astuple(), kernel)
        gpflow.set_trainable(gpr.likelihood, False)
        gpflow.set_trainable(gpr.kernel.kernels[0].variance, False)
        gpflow.set_trainable(gpr.kernel.kernels[1].period, False)
        model = GaussianProcessRegression(gpr, num_kernel_samples=100)
        model.optimize(dataset)

        # 3
        match self.config.acquisition_function:
            case "iv":
                logging.info(
                    "Using the ImprovementVariance acquisition function")
                rule = EfficientGlobalOptimization(
                    ImprovementVariance(threshold=5))
            case "ei":
                logging.info(
                    "Using the ExpectedImprovement acquisition function")
                rule = EfficientGlobalOptimization(ExpectedImprovement())
            case _:
                raise ValueError(
                    f"Unknown acquisition function {self.config.acquisition_function}"
                )

        # 4
        ask_only = AskTellOptimizerNoTraining(self.search_space, dataset,
                                              model, rule)
        new_point = ask_only.ask()
        return new_point.tolist()[0]
