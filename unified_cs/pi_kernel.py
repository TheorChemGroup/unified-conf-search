from scipy.optimize import minimize
import gpflow
import numpy as np
import tensorflow as tf

import json
import logging
import sqlite3
from typing import Callable, Union, Optional

import networkx as nx

from .utils import (
    compute_graph_checksum,
    same_graphs,
    pyd_dataclass,
)

CURRENT_COEFS_VERSION = 1


@pyd_dataclass
class PotentialCoefs:
    coefs: np.ndarray
    version: int

    @classmethod
    def empty(cls) -> 'PotentialCoefs':
        return PotentialCoefs(
            coefs=np.zeros(7),
            version=CURRENT_COEFS_VERSION,
        )


def pes_np(x: np.ndarray, A1: float, A2: float, A3: float, F1: float,
           F2: float, F3: float, C: float) -> np.ndarray:
    """Function that models PES of a single dihedral"""
    return (A1 * np.cos(x + F1) + A2 * np.cos(2 * x + F2) +
            A3 * np.cos(3 * x + F3) + C)


def pes_tf(x: tf.Tensor, A1: float, A2: float, A3: float, F1: float, F2: float,
           F3: float, C: float) -> tf.Tensor:
    """TensorFlow version of PES function modeling a single dihedral"""
    return (A1 * tf.cos(x + F1) + A2 * tf.cos(2.0 * x + F2) +
            A3 * tf.cos(3.0 * x + F3) + C)


def get_potential_minima(params: PotentialCoefs) -> list[tuple[float, float]]:
    filled_params = lambda x: pes_tf(x, *params.coefs)

    def tf_function_and_grad(x_np):
        x = tf.Variable(x_np[0], dtype=tf.float64)
        with tf.GradientTape() as tape:
            y = filled_params(x)
        grad = tape.gradient(y, x)
        return float(y.numpy()), np.array([grad.numpy()], dtype=np.float64)

    initial_guesses = np.linspace(-np.pi, np.pi, 5)
    minima = []

    for x0 in initial_guesses:
        res = minimize(tf_function_and_grad,
                       x0=np.array([x0]),
                       method='BFGS',
                       jac=True,
                       options={
                           'gtol': 1e-8,
                           'disp': False
                       })

        if res.success:
            x_min = res.x[0] % (2 * np.pi)
            y_min = res.fun

            if not any(np.isclose(x_min, xm, atol=1e-4) for xm, _ in minima):
                minima.append((x_min, y_min))

    minima.sort()
    ic(minima)
    return minima


def approximate_dihedral_pes(x: np.ndarray, y: np.ndarray) -> PotentialCoefs:
    """
    x - observed points [N, inp_dims]
    y - observed signal [N]
    returns [7, inp_dims] array of coefs
    """
    from scipy.optimize import curve_fit
    coefs, cov_matrix = curve_fit(
        pes_np,
        x,
        y,
        p0=np.random.uniform(0.0, 1.0, size=7),
        maxfev=50000,
    )
    return PotentialCoefs(coefs, version=CURRENT_COEFS_VERSION)


class CoefficientStorage:

    @pyd_dataclass
    class Entry:
        checksum: str
        version: int
        nodes: list[int]
        edges: list[tuple[int, int]]
        node_attrs: list[dict[str, str | int | float]]
        edge_attrs: list[dict[str, str | int | float]]
        params: Optional[np.ndarray] = None

        def to_raw(self) -> tuple[str, int, str, str, str, str, str]:
            assert self.params is not None
            return (
                self.checksum,
                self.version,
                json.dumps(self.nodes),
                json.dumps(self.edges),
                json.dumps(self.node_attrs),
                json.dumps(self.edge_attrs),
                json.dumps(self.params.tolist()),
            )

        @classmethod
        def from_raw(
            cls,
            raw_entry: tuple[str, ...],
        ) -> 'CoefficientStorage.Entry':
            return cls(
                checksum=raw_entry[0],
                version=raw_entry[1],
                nodes=json.loads(raw_entry[2]),
                edges=json.loads(raw_entry[3]),
                node_attrs=json.loads(raw_entry[4]),
                edge_attrs=json.loads(raw_entry[5]),
                params=np.array(json.loads(raw_entry[6])),
            )

        def build_graph(self) -> nx.Graph:
            result = nx.Graph()
            result.add_nodes_from(
                (n, attrs) for n, attrs in zip(self.nodes, self.node_attrs))
            result.add_edges_from(
                (u, v, attrs)
                for (u, v), attrs in zip(self.edges, self.edge_attrs))
            return result

        def to_coefs_object(self) -> PotentialCoefs:
            return PotentialCoefs(
                coefs=self.params,
                version=self.version,
            )

    def __init__(
        self,
        db_path: str,
        node_attrs: list[str],
        edge_attrs: list[str],
    ):
        self.node_attrs = [*node_attrs, 'dihedral_position']
        self.edge_attrs = edge_attrs

        self.db_path = db_path
        self.connection = sqlite3.connect(self.db_path)
        self.cursor = self.connection.cursor()
        self.init_table()

    def init_table(self) -> None:
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS graph_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            graph_checksum TEXT,
            version INTEGER,
            nodes TEXT,
            edges TEXT,
            node_attrs TEXT,
            edge_attrs TEXT,
            params TEXT
        )
        ''')
        self.connection.commit()

    def build_entry_object(
        self,
        graph: nx.Graph,
        dihedral_atoms: tuple[int, int, int, int],
        version: int = CURRENT_COEFS_VERSION,
    ) -> Entry:
        cur_graph = graph.copy()
        for node, data in cur_graph.nodes(data=True):
            data['dihedral_position'] = -1
        for i, atom in enumerate(dihedral_atoms):
            cur_graph.nodes[atom]['dihedral_position'] = i
        graph_checksum = compute_graph_checksum(cur_graph, self.node_attrs,
                                                self.edge_attrs)

        return CoefficientStorage.Entry(
            checksum=graph_checksum,
            version=version,
            nodes=[n for n in cur_graph.nodes],
            edges=[e for e in cur_graph.edges],
            node_attrs=[{
                k: v
                for k, v in data.items() if k in self.node_attrs
            } for n, data in cur_graph.nodes(data=True)],
            edge_attrs=[{
                k: v
                for k, v in data.items() if k in self.edge_attrs
            } for a, b, data in cur_graph.edges(data=True)],
        )

    def store_params(
        self,
        graph: nx.Graph,
        dihedral_atoms: tuple[int, int, int, int],
        params: PotentialCoefs,
    ) -> None:
        new_entry = self.build_entry_object(
            graph,
            dihedral_atoms,
            version=params.version,
        )
        if self.find_params_for_entry(new_entry) is not None:
            return None

        new_entry.params = params.coefs
        self.cursor.execute(
            '''
            INSERT INTO graph_data
            (graph_checksum, version, nodes, edges, node_attrs, edge_attrs, params)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''',
            new_entry.to_raw(),
        )
        self.connection.commit()

    def same_entries(self, first_entry: Entry, second_entry: Entry) -> bool:
        first_graph = first_entry.build_graph()
        second_graph = second_entry.build_graph()
        return same_graphs(first_graph, second_graph, self.node_attrs,
                           self.edge_attrs)

    def find_params(
        self,
        graph: nx.Graph,
        dihedral_atoms: tuple[int, int, int, int],
        version: int = CURRENT_COEFS_VERSION,
    ) -> Optional[PotentialCoefs]:
        search_entry = self.build_entry_object(
            graph,
            dihedral_atoms,
            version=version,
        )
        return self.find_params_for_entry(search_entry)

    def find_params_for_entry(self,
                              search_entry: Entry) -> Optional[PotentialCoefs]:
        self.cursor.execute(
            'SELECT * FROM graph_data WHERE graph_checksum = ? AND version = ?',
            (search_entry.checksum, search_entry.version))
        existing_entries = [
            CoefficientStorage.Entry.from_raw(raw_entry[1:])
            for raw_entry in self.cursor.fetchall()
        ]

        accepted_entries = [
            check_entry for check_entry in existing_entries
            if self.same_entries(search_entry, check_entry)
        ]

        if len(accepted_entries) < len(existing_entries):
            logging.warning("WOW! Detected a hash collision!")

        if len(accepted_entries) == 0:
            return None
        elif len(accepted_entries) > 1:
            logging.warning(
                "Multiple potentials found! Using an arbitrary one")

        return accepted_entries[0].to_coefs_object()


class FullPotentialFunction():

    def __init__(self, mean_func_coefs: list[PotentialCoefs]) -> None:
        self.mean_func_coefs = mean_func_coefs

    @tf.function
    def __call__(self, X: tf.Tensor) -> tf.Tensor:
        return tf.stack([
            pes_tf(X[:, dim], *(self.mean_func_coefs[dim].coefs))
            for dim in range(len(self.mean_func_coefs))
        ],
                        axis=1)


class TransformKernel(gpflow.kernels.Kernel):
    """The transform kernel. Requieres transform function and base kernel
            k(x, y) = base_kernel(f(x), f(y))
       where base_kernel - another kernel function.
    """

    def __init__(
        self,
        f: Callable[[tf.Tensor], tf.Tensor],
        base_kernel: gpflow.kernels.Kernel,
    ) -> None:
        super().__init__()

        self.f = f
        self.base_kernel = base_kernel

    def K(
        self,
        X1: tf.Tensor,
        X2: Union[tf.Tensor, None] = None,
    ) -> tf.Tensor:

        if X2 is None:
            X2 = X1

        return self.base_kernel.K(self.f(X1), self.f(X2))

    def K_diag(self, X: tf.Tensor) -> tf.Tensor:
        return self.base_kernel.K_diag(self.f(X))
