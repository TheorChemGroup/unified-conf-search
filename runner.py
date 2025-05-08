"""This is the runner of unified confsearch method"""
import ringo
import vf3py
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering

from unified_cs import (
    BOPredictor,
    Dof,
    ConformationInfo,
    CoefficientStorage,
    PotentialCoefs,
    pes_np,
    approximate_dihedral_pes,
    load_config,
    GlobalConfig,
    InitialGeneratorConfig,
    run_grid_search,
    get_potential_minima,
    LEVELS_OF_THEORY,
    MoleculeOptimizer,
)
from unified_cs.utils import (
    rdmol_to_graph,
    graph_to_rdmol,
    build_model_dihedral,
    get_equivalence_classes,
    wrap_angle,
    optimize_rdmol,
    get_xyzs_from_rdmol,
    get_unique_dihedrals,
    get_boring_dihedrals,
    temp_config_copy,
)

import multiprocessing as mp
import tqdm
import json
import random
import tempfile
import sys
import time
import math
import os
import logging
from typing import Optional
from pydantic.dataclasses import dataclass
from pydantic import ConfigDict

from icecream import install

install()

logging.getLogger().setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
# formatter = logging.Formatter('%(name)s:%(levelname)s: %(message)s')
formatter = logging.Formatter(
    '%(name)s:%(levelname)s:%(asctime)s: %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ConfPair:
    opt: ConformationInfo
    preopt: Optional[ConformationInfo] = None


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ConcreteScanPoint:
    dihedral: tuple[int, int, int, int]
    set_value: float
    actual_value: float
    energy: float
    xyz: np.ndarray
    dofs: dict[int, float]


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ModelDihedral:
    g: nx.Graph
    dof_indices: list[int]
    rotate_dihedral: tuple[int, int]


SCAN_DEBUG_DIR = "./temp"
COEF_DB_PATH = 'coef.db'

ADDITIONAL_FIXED_DIHEDRALS = {
    'pdb_1NWX': [],
    'pdb_2QZK': [(37, 39)],
    'csd_MIWTER': [],
    'pdb_2C6H': [],
    'pdb_2IYA': [],
    'SpnF_normalTS': [],
    'SpnF_altTS': [],
    'triperoxideAcetone': [],
    'triperoxideAcetone1': [],
    'triperoxideAcetone2': [],
}

ADD_BONDS = { # Indexing from 0
    'pdb_1NWX': [],
    'pdb_2QZK': [],
    'csd_MIWTER': [],
    'pdb_2C6H': [],
    'pdb_2IYA': [],
    'SpnF_normalTS': [(12, 13), (17, 18)],
    'SpnF_altTS': [(14, 15), (9, 11)],
    'triperoxideAcetone': [(23, 32), (33, 34)],
    'triperoxideAcetone1': [(23, 32), (33, 34)],
    'triperoxideAcetone2': [(23, 32), (33, 34)],
}
# REQUEST_FREE = { # Indexing from 1
#     'pdb_1NWX': [],
#     'pdb_2QZK': [],
#     'csd_MIWTER': [],
#     'pdb_2C6H': [],
#     'pdb_2IYA': [],
#     'SpnF_normalTS': [],
#     'SpnF_altTS': [(7, 8), (7, 6), (1, 5), (4, 5)],
# }


def some_four_atoms(
    graph: nx.Graph,
    edge: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Finds four nodes a-edge[0]-edge[1]-b in the graph"""
    u, v = edge
    assert graph.has_edge(u, v)

    for a in graph.neighbors(u):
        if a != v:
            for b in graph.neighbors(v):
                if b != u:
                    return (a, u, v, b)
    raise Exception(f"Unable to find a proper dihedral for edge {edge}")


class ConfsearchWrapper:

    def __init__(self, config: GlobalConfig, shortcut: Optional[dict] = None):
        # Initialize IK
        self.config = config
        self.molname = os.path.basename(self.config.input_sdf).replace(
            '.sdf', '')

        self.rdmol = Chem.MolFromMolFile(self.config.input_sdf, removeHs=False)
        Chem.SanitizeMol(self.rdmol)
        Chem.Kekulize(self.rdmol)
        boring_dihedrals = get_boring_dihedrals(rdmol_to_graph(self.rdmol))

        self.bondset = ic([(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                           for bond in self.rdmol.GetBonds()]) + ic(
                               self.config.added_bonds)
        self.bondset = {tuple(sorted(x)) for x in self.bondset}
        # raise Exception(self.bondset)

        if shortcut is None:
            if len(self.config.add_bonds) > 0:
                new_mol = self._get_honest_rdmol()
                with tempfile.NamedTemporaryFile(mode='w+',
                                                 delete=False) as temp:
                    temp.write(Chem.MolToMolBlock(new_mol))
                    temp.flush()
                    ringo_sdf = temp.name
            else:
                ringo_sdf = self.config.input_sdf
            logging.info(f"Init ringo from {ringo_sdf}")

            ringo.set_radius_multiplier(0.6, "ringo")
            self.iksolver = ringo.Molecule(
                sdf=ringo_sdf,
                request_fixed=self.config.request_fixed +
                [(i + 1, j + 1) for i, j in boring_dihedrals],
                # request_free=REQUEST_FREE[self.molname],
            )
            # self.iksolver = ringo.Molecule(sdf=self.config.input_sdf,
            #                                request_free=[(17, 18), (2, 28),
            #                                              (5, 6)])
            self.ring_atoms = self.iksolver.get_biggest_ringfrag_atoms()
            self.dofs_list, self.dofs_values = self.iksolver.get_ps()
        else:
            self.dofs_list = shortcut['dofs_list']

        # Initialize BO
        if shortcut is None:
            bo_dofs = [
                Dof(
                    lowest_value=0.0,
                    highest_value=2 * np.pi,
                    atoms=dof_atoms,
                ) for dof_atoms in self.dofs_list
            ]
            self.bo_pred = BOPredictor(bo_dofs, self.config.bo_config)

        self.distance_constraints = [
            (a, b, self.rdmol.GetConformer().GetAtomPosition(a).Distance(
                self.rdmol.GetConformer().GetAtomPosition(b)))
            for a, b in self.config.add_bonds
        ]

        self.optimizer: MoleculeOptimizer = LEVELS_OF_THEORY[
            self.config.leveloftheory._program_name](
                self.config.leveloftheory,
                distance_constraints=self.distance_constraints)

        if (self.config.ik_config.mode == 'grid' or
                self.config.bo_config.use_physics_kernel) and shortcut is None:
            self.prepare_dihedral_potentials()

        self.topo_p = ringo.Confpool()
        self.topo_p.atom_symbols = [
            atom.GetSymbol() for atom in self.rdmol.GetAtoms()
        ]

    def _get_honest_rdmol(self):
        if len(self.config.add_bonds) == 0:
            return self.rdmol

        emol = Chem.EditableMol(self.rdmol)
        for a, b in self.config.add_bonds:
            emol.AddBond(a, b, order=Chem.BondType.SINGLE)
        new_mol = emol.GetMol()
        return new_mol

    def topology_valid(self, xyzs: np.ndarray, add_bonds=[]) -> bool:
        for i in range(len(self.topo_p)):
            del self.topo_p[i]

        self.topo_p.include_from_xyz(xyzs, "")

        try:
            self.topo_p.generate_connectivity(0,
                                              mult=1.3,
                                              add_bonds=[(i + 1, j + 1)
                                                         for i, j in add_bonds
                                                         ])
        except RuntimeError as e:  # In case of disconnected topology
            logging.info(f"Topology invalid due to: {e}")
            return False

        found_bonds = {(a, b) if a < b else (b, a)
                       for a, b in self.topo_p.get_connectivity().edges}
        if found_bonds == self.bondset:
            return True
        else:
            logging.info(
                f"Topology invalid due to extra bonds: {found_bonds.difference(self.bondset)} "
                f"and missing bonds {self.bondset.difference(found_bonds)}")
            return False

    def extract_dofs(self, xyzs: np.ndarray) -> np.ndarray:
        p = ringo.Confpool()
        p.include_from_xyz(xyzs, "")
        result = self._extract_dofs_from_molproxy(p[0])
        return result

    def _extract_dofs_from_molproxy(self, m: ringo.MolProxy) -> np.ndarray:
        dof_values = [
            m.z(*[i + 1 for i in dof_atoms]) * ringo.DEG2RAD
            for dof_atoms in self.dofs_list
        ]
        return np.array(dof_values)

    def build_initial_ensemble(self) -> list[ConformationInfo]:
        match self.config.ik_config.mode:
            case "random":
                return self._build_random_ensemble(self.config.ik_config)
            case "mcr":
                return self._build_mcr_ensemble(self.config.ik_config)
            case "grid":
                return self._build_grid_ensemble(self.config.ik_config)
            case "existing":
                return self._load_existing_ensemble(self.config.ik_config)
            case _:
                raise ValueError(
                    f"Unknown initialization mode: {self.config.ik_config.mode}"
                )

    def _build_mcr_ensemble(
            self, ik_config: InitialGeneratorConfig) -> list[ConformationInfo]:

        res_p = ringo.Confpool()

        cs_settings = ringo.ConfsearchInput(
            molecule=self.iksolver,
            pool=res_p,
            termination_conditions=ringo.TerminationConditions(timelimit=30),
            rmsd_settings='default',
            geometry_validation={
                "ringo": {
                    "bondlength": 0.05,
                    "valence": 3.0,
                    "dihedral": 3.0,
                }
            },
            nthreads=1,
        )
        ringo.run_confsearch(cs_settings)
        assert res_p.size > 0, "No conformers, that's sus"
        res_p.save_xyz("tempensemble_mcr.xyz")
        return self.initialize_from_raw_confpool(res_p)

    def _build_random_ensemble(
            self, ik_config: InitialGeneratorConfig) -> list[ConformationInfo]:
        initial_conformations: list[ConformationInfo] = []
        while len(initial_conformations) < ik_config.num_iterations:
            logging.info(
                f"New IK trial for initial ensemble, iter #{len(initial_conformations)} out of {ik_config.num_iterations}"
            )
            pair_to_process = self.process_guess(
                [
                    random.uniform(-math.pi, math.pi)
                    for _ in range(len(self.dofs_list))
                ],
                process_corrupted=False,
            )

            if pair_to_process is not None:
                initial_conformations += pair_to_process
        return initial_conformations

    def _build_grid_ensemble(
            self, ik_config: InitialGeneratorConfig) -> list[ConformationInfo]:

        assert os.path.isdir(
            ik_config.cache_path), f"No such directory {ik_config.cache_path}"
        # cache_xyzpath = "/home/knvvv/bo/ringo_ensembles/macromodel_pdb1NWX.xyz"
        cache_xyzpath = os.path.join(ik_config.cache_path,
                                     f"{self.molname}_gridens.xyz")

        if not os.path.isfile(cache_xyzpath):
            preferred_dihedral_values = [[
                dihedral * ringo.RAD2DEG
                for dihedral, energy in get_potential_minima(coefs)
            ] for coefs in self.dof_params]

            p_initial = run_grid_search(
                mol=self.iksolver,
                dihedral_values=preferred_dihedral_values,
                config=ik_config,
                input_sdf=self.config.input_sdf,
            )

            # p_initial.generate_connectivity(0,
            #                                 mult=1.3,
            #                                 ignore_elements=['HCarbon'])
            # p_initial.generate_isomorphisms()
            # p_initial.rmsd_filter(0.3, num_threads=16, print_status=True)
            p_initial.save_xyz(cache_xyzpath)
            logging.info(f"Grid search results are cached in {cache_xyzpath}")
        else:
            p_initial = ringo.Confpool()
            p_initial.include_from_file(cache_xyzpath)
            logging.info(f"Loaded from {cache_xyzpath}")

        ensemble = self.initialize_from_raw_confpool(p_initial)
        # raise Exception("E")
        return ensemble

    def initialize_from_raw_confpool(
            self, p: ringo.Confpool) -> list[ConformationInfo]:
        logging.info("Extracting dofs")
        dofs = np.array(
            [self._extract_dofs_from_molproxy(m) for m in tqdm.tqdm(p)])

        if len(p) > 1:
            logging.info("Building dist matrix")
            distance_matrix = pdist(
                dofs,
                metric=lambda a, b: np.max(np.abs((a - b) % (2 * np.pi))),
            )
            square_dist = squareform(distance_matrix)
            logging.info("Running clustering")
            clustering = AgglomerativeClustering(
                metric='precomputed',
                linkage='complete',
                distance_threshold=5.0 * ringo.DEG2RAD,
                n_clusters=None,
            )
            logging.info("Generating cluster labels")
            labels = clustering.fit_predict(square_dist)

            from collections import defaultdict
            label_to_indices = defaultdict(list)
            for idx, label in enumerate(labels):
                label_to_indices[label].append(idx)
            label_to_indices = dict(label_to_indices)
            logging.debug(
                f"Got the following conformer labelling: {label_to_indices}")
        else:
            label_to_indices = {1: [0]}

        initial_conformations: list[ConformationInfo] = []
        for cluster_indices in tqdm.tqdm(label_to_indices.values()):
            pair_to_process = self._postprocess_solutions(
                [p[i].xyz for i in cluster_indices],
                dofs=[dofs[i] for i in cluster_indices],
                save_preopt=False,
            )
            if pair_to_process is None:
                continue
            initial_conformations.append(pair_to_process.opt)
        return initial_conformations

    def process_guess(
        self,
        dofs: list[float],
        process_corrupted: bool,
    ) -> list[ConformationInfo]:
        assert len(dofs) == len(self.dofs_list)
        for i, newvalue in enumerate(dofs):
            self.dofs_values[i] = newvalue

        result = self.iksolver.prepare_solution_iterator()
        logging.info(f"Success = {result==0}")

        if result != 0 and process_corrupted:
            logging.warning(f"IK FAILED TO GENERATE: result = {result}")
            return [
                ConformationInfo(
                    xyzs=None,
                    dofs=np.array(dofs),
                    minimum=False,
                    corrupted=True,
                    energy=None,
                )
            ]
        elif result != 0 and not process_corrupted:
            logging.warning(f"IK FAILED TO GENERATE: result = {result}")
            return None

        sol_list = self.iksolver.get_solutions_list()
        logging.info(f"Got {len(sol_list)} solutions")

        pair_to_process = self._postprocess_solutions(
            sol_list, dofs=[dofs for i in range(len(sol_list))])
        if pair_to_process is None:
            return []

        if pair_to_process.preopt is not None:
            return [
                pair_to_process.preopt,
                pair_to_process.opt,
            ]
        else:
            return [pair_to_process.opt]

    @staticmethod
    def _postprocess_in_thread(args):
        config, dofs_list, sol, dofs, save_preopt = args
        obj = ConfsearchWrapper(config, shortcut={'dofs_list': dofs_list})
        result = obj._postpocess_single(sol, dofs, save_preopt)
        return result

    def _postpocess_single(self,
                           input_xyzs: np.ndarray,
                           dofs: Optional[list[float]] = None,
                           save_preopt=True) -> ConfPair:
        if dofs is None:
            original_dofs = self.extract_dofs(input_xyzs)
        else:
            original_dofs = dofs

        try:
            preopt_geom, preopt_energy = self.optimizer.optimize(
                mol=self.rdmol,
                xyzs=input_xyzs,
                dihedral_constraints=[
                    (atoms, value)
                    for atoms, value in zip(self.dofs_list, original_dofs)
                ])
        except RuntimeError:
            return None

        save_preopt = save_preopt and preopt_energy is not None
        if save_preopt:
            preopt_dofs = self.extract_dofs(preopt_geom)

        try:
            optimized_xyzs, optimized_energy = self.optimizer.optimize(
                self.rdmol, preopt_geom)
        except RuntimeError:
            return None

        optimized_dofs = self.extract_dofs(optimized_xyzs)

        return ConfPair(
            preopt=ConformationInfo(
                xyzs=preopt_geom,
                dofs=preopt_dofs,
                energy=preopt_energy,
                minimum=False,
            ) if save_preopt else None,
            opt=ConformationInfo(
                xyzs=optimized_xyzs,
                dofs=optimized_dofs,
                energy=optimized_energy,
                minimum=True,
            ),
        )

    def _postprocess_solutions(
        self,
        conf_coords: list[np.ndarray],
        dofs: Optional[list[list[float]]] = None,
        save_preopt=True,
    ) -> ConfPair:
        if dofs is None:
            dofs = [None for i in range(len(conf_coords))]
        tasks = [(self.config, self.dofs_list, start_xyzs, cur_dofs,
                  save_preopt)
                 for start_xyzs, cur_dofs in zip(conf_coords, dofs)]
        with mp.Pool(processes=mp.cpu_count() - 2) as pool:
            confs = list(
                pool.map(ConfsearchWrapper._postprocess_in_thread, tasks))

        # confs: list[ConfPair] = []
        # for conf_index, input_xyzs in enumerate(conf_coords):
        #     if dofs is None:
        #         cur_dofs = None
        #     else:
        #         cur_dofs = dofs[conf_index]
        #     confpair = self._postpocess_single(input_xyzs,
        #                                        cur_dofs,
        #                                        save_preopt=save_preopt)

        #     confs.append(confpair)

        logging.info("Summary of generated pairs:")
        for i, pair in enumerate(confs, start=1):
            if pair is None:
                preopt_summ = "None (all)"
                opt_summ = "None (all)"
            else:
                if pair.preopt is None:
                    preopt_summ = "None"
                else:
                    preopt_summ = pair.preopt.energy
                if pair.opt is None:
                    opt_summ = "None"
                else:
                    opt_summ = pair.opt.energy
            logging.info(f"{i}) preopt={preopt_summ} opt={opt_summ}")

        logging.info("Filtering nones and validating")
        confs = [
            conf for conf in confs if conf is not None and self.topology_valid(
                conf.opt.xyzs, add_bonds=self.config.added_bonds)
        ]

        if len(confs) == 0:
            return None

        best_pair = min(confs, key=lambda pair: pair.opt.energy)
        return best_pair

    def _load_existing_ensemble(
            self, ik_config: InitialGeneratorConfig) -> list[ConformationInfo]:
        p = ringo.Confpool()
        logging.info(f"Loading XYZ file {ik_config.existing_path}")
        p.include_from_file(ik_config.existing_path)
        logging.info(f"Done loading XYZ file {ik_config.existing_path}")

        def get_energy(rawinfo):
            if 'absenergy' in rawinfo:
                return rawinfo['absenergy']
            elif 'gfnff_ener' in rawinfo:
                return rawinfo['gfnff_ener']
            else:
                raise Exception("Cannot get energy")

        result = []
        for m in p:
            rawinfo = json.loads(m.descr)
            info = ConformationInfo(
                xyzs=m.xyz,
                dofs=self._extract_dofs_from_molproxy(m),
                energy=get_energy(rawinfo),
                minimum=rawinfo['minimum'] if 'minimum' in rawinfo else True,
            )
            result.append(info)
        return result

    def list_to_confpool(
        self,
        conflist: list[ConformationInfo],
        minima_only=True,
    ) -> ringo.Confpool:
        # assert all(not x.corrupted for x in conflist)
        min_energy = min(
            [conf.energy for conf in conflist if not conf.corrupted])
        data = [{
            'relenergy': (conf.energy - min_energy),
            'absenergy': conf.energy,
            'minimum': conf.minimum,
            'source': conf.source,
        } for conf in conflist if not conf.corrupted]
        ids = [i for i, conf in enumerate(conflist) if not conf.corrupted]
        descriptions = [json.dumps(d) for d in data]

        logging.info(
            f"Num bo origins (1) {len([x for x in data if x['source'] == 'bo'])}"
        )
        logging.info(
            f"bo origins (1) {[x for x in data if x['source'] == 'bo']}")

        p = ringo.Confpool()
        for i, descr in zip(ids, descriptions):
            conf = conflist[i]
            if conf.corrupted:
                logging.info(f"Skip {descr}")
                continue
            if minima_only and not conf.minimum:
                logging.info(f"Skip {descr}")
                continue
            p.include_from_xyz(conf.xyzs, descr)

        logging.info(
            f"Num bo origins (2) {len([m for m in p if json.loads(m.descr)['source'] == 'bo'])}"
        )
        logging.info(
            f"bo origins (2) {[m.descr for m in p if json.loads(m.descr)['source'] == 'bo']}"
        )
        p.atom_symbols = [atom.GetSymbol() for atom in self.rdmol.GetAtoms()]
        return p

    def build_bondgraphs(self):
        graph = rdmol_to_graph(self.rdmol)

        bondgraphs = {
            dof_id: build_model_dihedral(graph, dof_atoms)
            for dof_id, dof_atoms in enumerate(self.dofs_list)
        }
        bondgraphs = {
            dof_id: graph
            for dof_id, graph in bondgraphs.items() if graph is not None
        }

        node_match = lambda na, nb: (na['symbol'] == nb['symbol'] and na[
            'extra'] == nb['extra'])
        edge_match = lambda ea, eb: (ea['bondorder'] == eb['bondorder'] and ea[
            'scanned'] == eb['scanned'])

        # unique_bondgraph_indices: index of cluster => index of class representative in bondgraphs
        # bondgraphs_classes: index in bondgraphs => index of cluster
        unique_bondgraph_indices, bondgraphs_classes = get_equivalence_classes(
            bondgraphs,
            eq=lambda graph_a, graph_b: (graph_a is not None) and
            (graph_b is not None) and vf3py.are_isomorphic(
                graph_a, graph_b, node_match=node_match, edge_match=edge_match
            ),
        )

        # atommap_to_class: index in bondgraphs => {atom label in the bondgraph => atom label in class representative}
        atommap_to_class = {
            dof_index:
            vf3py.are_isomorphic(
                bondgraphs[dof_index],
                bondgraphs[unique_bondgraph_indices[class_index]],
                node_match=node_match,
                edge_match=edge_match,
                # We already know that graphs are isomorphic
                get_mapping=True)[1]
            for dof_index, class_index in bondgraphs_classes.items()
        }
        atommap_to_class: dict[int, dict[int, int]] = {
            dof_index: {
                source: target
                for source, target in mapping.items()
                if not bondgraphs[dof_index].nodes[source]['extra']
            }
            for dof_index, mapping in atommap_to_class.items()
        }

        unique_bondgraphs: list[nx.Graph] = [
            bondgraphs[i] for i in unique_bondgraph_indices
        ]

        unique_bondgraphs_relabeling_maps = []
        for cur_graph in unique_bondgraphs:
            relabel_map = {
                old_label: new_label
                for new_label, old_label in enumerate(cur_graph.nodes)
            }
            nx.relabel_nodes(cur_graph, relabel_map, copy=False)
            unique_bondgraphs_relabeling_maps.append(relabel_map)

        # atommap_to_class: dof index => {atom label in molgraph => atom label in unique_bondgraph}
        atommap_to_class = {
            dof_index: {
                source:
                unique_bondgraphs_relabeling_maps[
                    bondgraphs_classes[dof_index]][old_target]
                for source, old_target in old_map.items()
            }
            for dof_index, old_map in atommap_to_class.items()
        }

        def get_rotated_dihedral(graph: nx.Graph) -> tuple[int, int]:
            rotated_edges = [(a, b) for a, b, data in graph.edges(data=True)
                             if data['scanned']]
            assert len(rotated_edges) == 1, (
                "Only one bond is allowed to be rotated per model molecule")
            return rotated_edges[0]

        models = [
            ModelDihedral(g=graph,
                          dof_indices=[
                              dof_index for dof_index, check_class_index in
                              bondgraphs_classes.items()
                              if check_class_index == graph_index
                          ],
                          rotate_dihedral=get_rotated_dihedral(graph))
            for graph_index, graph in enumerate(unique_bondgraphs)
        ]
        return bondgraphs_classes, models, atommap_to_class

    def prepare_dihedral_potentials(self):
        coef_storage = CoefficientStorage(
            COEF_DB_PATH,
            node_attrs=['symbol', 'charge', 'extra'],
            edge_attrs=['bondorder', 'scanned'])

        bondgraphs_classes, unique_bondgraphs, atommap_to_class = (
            self.build_bondgraphs())
        # ic(bondgraphs_classes, unique_bondgraphs, atommap_to_class)

        xx = lambda tup: tuple(i + 1 for i in tup)
        for bondgraph_index, modelmol in enumerate(unique_bondgraphs):
            rdmol = graph_to_rdmol(modelmol.g)
            Chem.SanitizeMol(rdmol)
            AllChem.EmbedMolecule(rdmol)
            Chem.MolToMolFile(
                rdmol, os.path.join(SCAN_DEBUG_DIR, f"{bondgraph_index}.mol"))
            logging.info(
                f"Rotating this bond in class #{bondgraph_index}: {xx(modelmol.rotate_dihedral)}"
            )
            for member_index in [
                    j for j, cur_class in bondgraphs_classes.items()
                    if cur_class == bondgraph_index
            ]:
                plus_one = {
                    k + 1: v + 1
                    for k, v in atommap_to_class[member_index].items()
                }
                logging.info(
                    f"Member #{member_index} ({xx(self.dofs_list[member_index])}) of class #{bondgraph_index}: {plus_one}"
                )

        dof_params_dict: dict[int, PotentialCoefs] = {}
        for dof_index, class_index in bondgraphs_classes.items():
            dof_atoms = self.dofs_list[dof_index]
            bondgraph_class = unique_bondgraphs[class_index]
            atom_mapping = atommap_to_class[dof_index]
            try:
                dof_model_atoms = tuple(atom_mapping[atom]
                                        for atom in dof_atoms)
                found_parameters = coef_storage.find_params(
                    graph=bondgraph_class.g,
                    dihedral_atoms=dof_model_atoms,
                )
            except KeyError:
                logging.info("Looks like someone have messed with molgraph")
                found_parameters = PotentialCoefs.empty()

            if found_parameters is not None:
                dof_params_dict[dof_index] = found_parameters
                logging.info(
                    f"Potential for dof #{dof_index} is pulled from cache")

        for dof_index in range(len(self.dofs_list)):
            if dof_index not in bondgraphs_classes:
                dof_params_dict[dof_index] = PotentialCoefs.empty()

        full_scan_info: dict[int, dict[int, ConcreteScanPoint]] = {}
        for bondgraph_index, modelmol in enumerate(unique_bondgraphs):
            if all(dof_index in dof_params_dict
                   for dof_index in modelmol.dof_indices):
                continue

            logging.info(
                f"Potential for dof(s) {modelmol.dof_indices} will be computed from scratch. bondgraph_index={bondgraph_index}"
            )

            assert bondgraph_index not in full_scan_info
            full_scan_info[bondgraph_index] = self.generate_scaninfo(
                modelmol,
                atommap_to_class,
                dump_scan_to_file=os.path.join(SCAN_DEBUG_DIR,
                                               f"{bondgraph_index}.xyz"),
            )

        for dof_index, class_index in bondgraphs_classes.items():
            if dof_index in dof_params_dict:
                continue

            dof_atoms = self.dofs_list[dof_index]
            bondgraph_class = unique_bondgraphs[class_index]
            assert dof_index in bondgraph_class.dof_indices, repr(
                bondgraph_class.dof_indices)

            cur_scan_info = full_scan_info[class_index]
            dihedral_list = np.array(
                [x.dofs[dof_index] for x in cur_scan_info.values()])

            energy_list = np.array([x.energy for x in cur_scan_info.values()])

            best_params, best_score = None, None
            num_tries = 0
            while num_tries < 10:
                try:
                    params = approximate_dihedral_pes(
                        dihedral_list,
                        energy_list,
                    )
                except:
                    logging.info("Failed for approximate. Trying again")
                    continue

                predicted_e = pes_np(dihedral_list, *params.coefs)
                cur_score = np.linalg.norm(np.abs(energy_list - predicted_e))

                if best_score is None or cur_score < best_score:
                    best_score = cur_score
                    best_params = params
                num_tries += 1

            predicted_e = pes_np(dihedral_list, *best_params.coefs)
            df = pd.DataFrame({
                'x': dihedral_list,
                'real': energy_list,
                'model': predicted_e,
            })
            df.to_csv(f'dof_{dof_index}.csv', index=False)
            dof_params_dict[dof_index] = best_params

            atom_mapping = atommap_to_class[dof_index]
            dof_model_atoms = tuple(atom_mapping[atom] for atom in dof_atoms)
            coef_storage.store_params(
                graph=bondgraph_class.g,
                dihedral_atoms=dof_model_atoms,
                params=params,
            )

        self.dof_params = [
            dof_params_dict[i] for i in range(len(dof_params_dict))
        ]

    def generate_scaninfo(
        self,
        modelmol: ModelDihedral,
        atommap_to_class: list[dict[int, int]],
        dump_scan_to_file: str | None = None,
    ):
        dihedral_atoms = some_four_atoms(modelmol.g, modelmol.rotate_dihedral)
        fixed_dihedrals = [
            atoms for atoms in get_unique_dihedrals(modelmol.g)
            if set(modelmol.rotate_dihedral) != {atoms[1], atoms[2]}
        ]

        get_dihedral = lambda m, atoms: (m.z(*[i + 1 for i in atoms]) * ringo.
                                         DEG2RAD)
        rdmol = graph_to_rdmol(modelmol.g)
        logging.info(f"Doing scan for {Chem.MolToSmiles(rdmol)}")
        Chem.SanitizeMol(rdmol)
        AllChem.EmbedMolecule(rdmol)
        current_geom, current_energy = optimize_rdmol(
            mol=rdmol, xyzs=get_xyzs_from_rdmol(rdmol))

        p = ringo.Confpool()
        p.include_from_xyz(current_geom, "")
        p.atom_symbols = [atom.GetSymbol() for atom in rdmol.GetAtoms()]

        def comment_last_structure(name: str, energy: float,
                                   excepted_dihedral: float):
            m = p[len(p) - 1]
            actual_dihedral = get_dihedral(m, dihedral_atoms)
            m.descr = name
            m['Energy'] = energy
            m['Actual dihedral'] = actual_dihedral
            m['Expected dihedral'] = excepted_dihedral

        def last_structure_to_scanpoint():
            m = p[len(p) - 1]
            return ConcreteScanPoint(
                dihedral=dihedral_atoms,
                actual_value=m['Actual dihedral'],
                set_value=m['Expected dihedral'],
                energy=m['Energy'],
                xyz=m.xyz,
                dofs={
                    dof_index:
                    get_dihedral(
                        m,
                        tuple(atommap_to_class[dof_index][atom_index]
                              for atom_index in self.dofs_list[dof_index]))
                    for dof_index in modelmol.dof_indices
                })

        # comment_last_structure("Initial", current_energy)
        start_value = get_dihedral(p[0], dihedral_atoms)
        fixed_values = [get_dihedral(p[0], atoms) for atoms in fixed_dihedrals]
        del p[0]

        cur_scan_info: dict[int, ConcreteScanPoint] = {}
        for step, cur_value in enumerate(
                np.arange(start_value, start_value + 2 * np.pi, np.pi / 180)):
            wrapped_value = wrap_angle(cur_value)
            current_geom, current_energy = optimize_rdmol(
                mol=rdmol,
                xyzs=current_geom,
                dihedral_constraints=[(dihedral_atoms, wrapped_value)] +
                [(atoms, value)
                 for atoms, value in zip(fixed_dihedrals, fixed_values)])
            p.include_from_xyz(current_geom, "")
            comment_last_structure(f"Step #{step}", current_energy,
                                   wrapped_value)
            assert step not in cur_scan_info
            cur_scan_info[step] = last_structure_to_scanpoint()

        lowest_energy = min(point.energy for point in cur_scan_info.values())
        for point in cur_scan_info.values():
            point.energy -= lowest_energy

        if dump_scan_to_file is not None:
            lowest_energy_pool = min(m['Energy'] for m in p)
            assert abs(lowest_energy - lowest_energy_pool) < 1e5
            p['Energy'] = lambda m: m['Energy'] - lowest_energy_pool
            p.descr = lambda m: json.dumps({
                'name':
                m.descr,
                'ener':
                m['Energy'],
                'dihedrals': [{
                    'atoms': dihedral_atoms,
                    'value': m['Actual dihedral'],
                }],
            })

            p.generate_isomorphisms(trivial=True)
            p.align_all_with(p[0], mirror_match=False)
            p.save_xyz(dump_scan_to_file)
        return cur_scan_info


if __name__ == "__main__":
    molname = sys.argv[1]
    with temp_config_copy('config.yaml', {
            'mol_name': molname,
            'molname': molname.replace('_', ''),
    }) as config_path:
        logging.info(f"Using temppath for completed config: {config_path}")
        config = load_config(config_path)
        logging.info(f"Completed config is {config}")
    config.request_fixed = ADDITIONAL_FIXED_DIHEDRALS[molname]
    config.add_bonds = ADD_BONDS[molname]
    config.added_bonds = ADD_BONDS[molname]

    config = load_config('config.yaml')

    if config.logfile is not None:
        file_handler = logging.FileHandler(config.logfile)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)

    cw = ConfsearchWrapper(config)
    current_conformations = cw.build_initial_ensemble()
    # current_conformations = [
    #     x for x in current_conformations
    #     if cw.topology_valid(x, add_bonds=config.add_bonds)
    # ]
    # cw.list_to_confpool(current_conformations).save_xyz(
    #     'triperoxideAcetone2.xyz')
    # raise Exception("E")
    logging.info("Validating the starting ensemble")
    assert all(
        cw.topology_valid(x.xyzs, add_bonds=config.add_bonds)
        for x in tqdm.tqdm(current_conformations))
    logging.info("Done validation of the starting ensemble")

    # cw.list_to_confpool(current_conformations).save_xyz('temp.xyz')
    logging.info(f"Starting ensemble size is {len(current_conformations)}")

    p = cw.list_to_confpool(current_conformations)
    p['list_index'] = lambda m: float(m.idx)
    p.generate_connectivity(0,
                            mult=1.3,
                            ignore_elements=['HCarbon'],
                            add_bonds=[(i + 1, j + 1)
                                       for i, j in config.add_bonds])
    p['ener'] = lambda m: json.loads(m.descr)['absenergy']
    p.sort('ener')
    ic(p.upper_cutoff('ener', 30.0))
    p.generate_isomorphisms()
    logging.info(f"Start RMSD filtering. Initial size={len(p)}")
    p.rmsd_filter(0.3, num_threads=16)
    p.filter(lambda m: m.idx < 1000)
    train_indices = sorted([int(m['list_index']) for m in p])
    logging.info(f"Starting train set size is {len(train_indices)}")

    def run_mtd(start_xyz) -> list[ConformationInfo]:
        mtd_coords = cw.optimizer.run_mtd(cw.rdmol, start_xyz)
        pnew = ringo.Confpool()
        pnew.atom_symbols = p.atom_symbols

        # for i, xyzs in enumerate(mtd_coords):
        #     opt_xyzs, energy = cw.optimizer.optimize(cw.rdmol, xyzs)
        #     pnew.include_from_xyz(opt_xyzs, f"{energy}")

        for i, xyzs in enumerate(mtd_coords):
            pnew.include_from_xyz(xyzs, "")

        tasks = [(cw.config, cw.dofs_list, m.xyz,
                  cw._extract_dofs_from_molproxy(m), False) for m in pnew]
        with mp.Pool(processes=mp.cpu_count() - 2) as pool:
            opt_results = list(
                pool.map(ConfsearchWrapper._postprocess_in_thread, tasks))
        opt_results = [conf for conf in opt_results if conf is not None]

        pnew.filter(lambda m: False)
        for pair in opt_results:
            pair: ConfPair
            if pair is None or not cw.topology_valid(
                    pair.opt.xyzs, add_bonds=config.add_bonds):
                continue
            pnew.include_from_xyz(pair.opt.xyzs, str(pair.opt.energy))

        if len(pnew) == 0:
            return []

        pnew.generate_connectivity(
            0,
            mult=1.3,
            ignore_elements=[
                node for node in range(pnew.natoms)
                if node not in cw.ring_atoms
            ],
            add_bonds=[(i + 1, j + 1) for i, j in config.added_bonds
                       if i in cw.ring_atoms and j in cw.ring_atoms])
        pnew.generate_isomorphisms()
        pnew.rmsd_filter(0.5)
        pnew.align_all_with(pnew[0])
        rmsd_matr = pnew.get_rmsd_matrix()
        logging.info(
            f"MTD iteration diversity: max RMSD = {np.max(rmsd_matr.flatten())}, mean RMSD = {np.mean(rmsd_matr.flatten())}"
        )
        return [
            ConformationInfo(
                xyzs=m.xyz,
                dofs=cw._extract_dofs_from_molproxy(m),
                energy=float(m.descr),
                corrupted=False,
                source='mtd',
                minimum=True,
            ) for m in pnew
        ]

    def get_filtered_pool(conformations: list[ConformationInfo], nice=False):
        p = cw.list_to_confpool(conformations)
        p.generate_connectivity(0, mult=1.3, ignore_elements=['HCarbon'])
        p.generate_isomorphisms()
        logging.info(f"Start RMSD filtering. Initial size={len(p)}")
        p.rmsd_filter(0.3, num_threads=16)
        # p['ener'] = lambda m: json.loads(m.descr)['relenergy']
        # p.sort('ener')
        # p.filter(lambda m: m.idx < 10)
        logging.info(f"Finish RMSD filtering. Final size={len(p)}")
        if nice:
            p.align_all_with(p[0], mirror_match=True)
        return p

    def should_train_on_new_geom(new_geom: ConformationInfo,
                                 list_index: int) -> bool:
        if new_geom.corrupted:
            return True

        cur_p_index = len(p)
        p.include_from_xyz(new_geom.xyzs, "")
        m = p[cur_p_index]
        m['list_index'] = float(list_index)
        m['ener'] = new_geom.energy
        min_rmsd = min(m.rmsd(test_m)[0] for test_m in p[:cur_p_index])
        ener = m['ener'] - min(m['ener'] for m in p[:cur_p_index])
        # train = True
        train = ener < 40.0 and min_rmsd > 0.1
        if not train:
            del p[cur_p_index]
        logging.info(
            f"New conf: min_rmsd={min_rmsd}, ener={ener} (train={train})")
        return train

    nothing_new_count = 0
    iteration_count = 0
    prev_unique_count = len(train_indices)
    while nothing_new_count < 20 and iteration_count < 100:
        logging.info(
            f"New iter #{iteration_count}. Size={len(current_conformations)} "
            f"nothing_new_count={nothing_new_count} "
            f"prev_unique_count={prev_unique_count}")
        start_iter_time = time.time()
        new_dofs = cw.bo_pred.predict_new_dofs(
            [current_conformations[i] for i in train_indices], cw.dof_params)
        end_iter_time = time.time()
        logging.info(f"GP iteration time {end_iter_time-start_iter_time}s")

        new_geoms = cw.process_guess(new_dofs, process_corrupted=True)
        mtd_new_geoms = []
        for i, new_geom in enumerate(new_geoms):
            cur_index = len(current_conformations)
            new_geom.source = "bo"
            current_conformations.append(new_geom)
            if should_train_on_new_geom(new_geom, cur_index):
                train_indices.append(cur_index)
                if not current_conformations[cur_index].corrupted:
                    logging.info(f"Running MTD for #{i}")
                    mtd_new_geoms += run_mtd(
                        current_conformations[cur_index].xyzs)
                    logging.debug(f"len(mtd_new_geoms) = {len(mtd_new_geoms)}")
                else:
                    logging.info(
                        f"Training on #{i}, but skipping MTD because it is corrupted"
                    )
            else:
                logging.info(f"Skipping #{i} due to RMSD or energy")

        for i, new_geom in enumerate(mtd_new_geoms):
            new_geom.source = "mtd"
            cur_index = len(current_conformations)
            current_conformations.append(new_geom)
            if should_train_on_new_geom(new_geom, cur_index):
                train_indices.append(cur_index)

        iteration_count += 1
        new_unique_count = len(train_indices)
        if new_unique_count == prev_unique_count:
            nothing_new_count += 1
        else:
            nothing_new_count = 0

        prev_unique_count = new_unique_count
        logging.info(
            f"End of iteration. nothing_new_count={nothing_new_count}, iteration_count={iteration_count}"
        )

    logging.info(
        f"Overall count of BO generated conformations = {len([m for m in current_conformations if m.source == 'bo'])}"
    )
    logging.info(
        f"Overall count of MTD generated conformations = {len([m for m in current_conformations if m.source == 'mtd'])}"
    )

    fullres_p = cw.list_to_confpool(current_conformations, minima_only=False)
    logging.info(
        f"Generated {len(fullres_p)} structures overall, including non-minima")
    fullres_p.save_xyz(f"{molname}_full.xyz")

    res_p = cw.list_to_confpool(current_conformations)
    logging.info(
        f"Generated {len(res_p)} structures overall, including non-minima")
    res_p.save_xyz(f"{molname}_result.xyz")

    logging.info("Normal termination")
