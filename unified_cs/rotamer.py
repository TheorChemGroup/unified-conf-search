import ringo
import vf3py
import networkx as nx
import numpy as np
import scipy.linalg

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from sage.all import *

import os
import itertools
import enum
import logging
from typing import Optional

from .config import RotamerAnalyzerConfig
from .utils import (
    rdmol_to_coord_matrix,
    rdmol_to_graph,
    gen_labels,
    determine_span,
    kabsch_rmsd,
    group_summary,
)


class NodeType(enum.Enum):
    ATOM = enum.auto()
    VALENCE = enum.auto()
    DIHEDRAL = enum.auto()


class EdgeType(enum.Enum):
    BOND = enum.auto()
    VALENCE_CENTER = enum.auto()
    VALENCE_SIDE = enum.auto()
    DIHEDRAL_A = enum.auto()
    DIHEDRAL_B = enum.auto()
    DIHEDRAL_C = enum.auto()
    DIHEDRAL_D = enum.auto()


VALENCE_EDGETYPES = (EdgeType.VALENCE_SIDE, EdgeType.VALENCE_CENTER,
                     EdgeType.VALENCE_SIDE)

DIHEDRAL_EDGETYPES = (EdgeType.DIHEDRAL_A, EdgeType.DIHEDRAL_B,
                      EdgeType.DIHEDRAL_C, EdgeType.DIHEDRAL_D)

TESTFILES_DIR = 'testfiles'
# INPUTFILE = 'methanol.xyz'
INPUTFILE = 'sf4.sdf'
# INPUTFILE = 'c2h2f2.sdf'

deg2m = lambda degs: scipy.linalg.block_diag(*[
    np.array([
        [np.cos(np.radians(deg)), -np.sin(np.radians(deg))],
        [np.sin(np.radians(deg)),
         np.cos(np.radians(deg))],
    ]) for deg in degs
])


def m2deg(matrix):
    angles = []
    n = matrix.shape[0]
    for i in range(0, n, 2):
        block = matrix[i:i + 2, i:i + 2]
        theta = np.rad2deg(np.arctan2(block[1, 0], block[0, 0]))
        angles.append(theta)
    return angles


class RotamerAnalyzer:
    TEMP_MOL_PATH = '.temp.mol'

    def __init__(
        self,
        molgraph: nx.Graph,
        coords: np.ndarray,
        config: RotamerAnalyzerConfig,
    ):
        self.molgraph = molgraph
        self.config = config

        assert os.path.isfile(self.TEMP_MOL_PATH)
        self.solver = ringo.Molecule(sdf=self.TEMP_MOL_PATH)
        os.remove(self.TEMP_MOL_PATH)
        self.dofs_atoms, self.dofs_values = self.solver.get_ps()

        self.p = ringo.Confpool()
        self.p.include_from_xyz(coords, "")
        self.p.atom_symbols = [
            self.molgraph.nodes[i]['symbol']
            for i in range(self.molgraph.number_of_nodes())
        ]
        self.structgraph = self.build_struct_graph()

        self.stereocenters: dict[int, Stereocenter] = {
            node:
            Stereocenter(
                coords,
                center_atom=node,
                nb_atoms=list(self.molgraph.neighbors(node)),
                atom_symbol=data['symbol'],
            )
            for node, data in self.molgraph.nodes(data=True)
            if len(list(self.molgraph.neighbors(node))) > 1
        }
        logging.debug(
            "The following stereocenters will be tested:\n" + '\n'.join(
                f"{i}) " + repr(c)
                for i, c in enumerate(self.stereocenters.values(), start=1)))

    @classmethod
    def from_xyz(cls, xyz_path: str,
                 config: RotamerAnalyzerConfig) -> 'RotamerAnalyzer':
        mol = Chem.MolFromXYZFile(xyz_path)
        assert mol is not None, f"Failed to read molecule from {xyz_path}"
        rdDetermineBonds.DetermineConnectivity(mol, useHueckel=True)
        Chem.MolToMolFile(mol, cls.TEMP_MOL_PATH)
        result = cls(rdmol_to_graph(mol), rdmol_to_coord_matrix(mol), config)
        return result

    @classmethod
    def from_sdf(cls, sdf_path: str,
                 config: RotamerAnalyzerConfig) -> 'RotamerAnalyzer':
        mol = Chem.MolFromMolFile(sdf_path, removeHs=False)
        assert mol is not None, f"Failed to read molecule from {sdf_path}"
        Chem.MolToMolFile(mol, cls.TEMP_MOL_PATH)
        result = cls(rdmol_to_graph(mol), rdmol_to_coord_matrix(mol), config)
        return result

    def build_struct_graph(self) -> nx.Graph:
        structgraph: nx.Graph = self.molgraph.copy()
        for atom, data in structgraph.nodes(data=True):
            data['type'] = NodeType.ATOM
        for a, b, data in structgraph.edges(data=True):
            data['type'] = EdgeType.BOND
        m = self.p[0]
        xx = lambda numbers: tuple(i + 1 for i in numbers)

        bonds = [(atom_a, atom_b)
                 for atom_a, atom_b, data in structgraph.edges(data=True)
                 if data['type'] == EdgeType.BOND]
        bondlengths = [m.l(*xx(atoms)) for atoms in bonds]
        bond_labels, bond_clusters = gen_labels(
            bondlengths, threshold=self.config.bondlength_threshold)
        for (atom_a, atom_b), label in zip(bonds, bond_labels):
            structgraph[atom_a][atom_b]['label'] = label
        logging.debug(f"Obtained bondlength clustering: {bond_clusters}")

        valence_angles = [(nb_a, center_atom, nb_b)
                          for center_atom in structgraph.nodes
                          for nb_a, nb_b in itertools.combinations(
                              structgraph.neighbors(center_atom), 2)]
        assert len(valence_angles) == len(
            set(valence_angles)), "Duplicates found?"
        valence_angle_values = [m.v(*xx(atoms)) for atoms in valence_angles]
        valence_labels, valence_clusters = gen_labels(
            valence_angle_values, threshold=self.config.valence_threshold)
        logging.debug(f"Obtained valence angle clustering: {valence_clusters}")

        for i, (atoms, label) in enumerate(zip(valence_angles,
                                               valence_labels)):
            new_node = f"valence_{i}"
            structgraph.add_node(new_node, type=NodeType.VALENCE, label=label)
            for atom, edgetype in zip(atoms, VALENCE_EDGETYPES):
                structgraph.add_edge(atom, new_node, type=edgetype)

        nonrotatable_bonds = [
            (node_a, node_b)
            for node_a, node_b, data in self.molgraph.edges(data=True)
            if data['bondorder'] != 1.0
        ]
        nonrotatable_dihedrals = {
            (node_a, node_b): [(nb_left, node_a, node_b, nb_right)
                               for nb_left, nb_right in itertools.product(
                                   self.molgraph.neighbors(node_a),
                                   self.molgraph.neighbors(node_b))
                               if (nb_left != node_b) and (nb_right != node_a)]
            for node_a, node_b in nonrotatable_bonds
        }
        nonrotatable_dihedrals_list = [
            dihedral_atoms
            for bond, dihedrals in nonrotatable_dihedrals.items()
            for dihedral_atoms in dihedrals
        ]
        nonrotatable_dihedral_values = [
            m.z(*xx(atoms)) for atoms in nonrotatable_dihedrals_list
        ]
        dihedral_labels, dihedral_clusters = gen_labels(
            nonrotatable_dihedral_values,
            threshold=self.config.fixed_dihedral_threshold,
            period=360.0)
        logging.debug(
            f"Obtained fixed dihedral clustering: {dihedral_clusters}")

        dihedral_node_count = 0
        for bond, dihedrals in nonrotatable_dihedrals.items():
            for dihedral_atoms in dihedrals:
                dihedral_node = f"dihedral_{dihedral_node_count}"
                dihedral_node_count += 1
                structgraph.add_node(
                    dihedral_node,
                    type=NodeType.DIHEDRAL,
                    label=dihedral_labels[nonrotatable_dihedrals_list.index(
                        dihedral_atoms)],
                )
                for node, edgetype in zip(dihedral_atoms, DIHEDRAL_EDGETYPES):
                    structgraph.add_edge(node, dihedral_node, type=edgetype)

        # ic([i for i in structgraph.nodes(data=True)])
        # ic([i for i in structgraph.edges(data=True)])

        return structgraph

    def get_rotamer_group(self) -> PermutationGroup:
        logging.info(
            "Generating automorphisms for augmented molgraph with "
            f"{self.structgraph.number_of_nodes()} nodes and {self.structgraph.number_of_edges()} edges"
        )
        perms = vf3py.get_automorphisms(
            self.structgraph,
            node_match=lambda a_node, b_node: a_node == b_node,
            edge_match=lambda a_edge, b_edge: a_edge == b_edge,
        )
        logging.info("Done generating graph automorphisms")

        num_atoms = self.molgraph.number_of_nodes()
        perms = [[match[i] for i in range(num_atoms)] for match in perms]

        # This potentially includes rigid motions and permutations that break stereochemistry
        full_group = PermutationGroup(perms,
                                      domain=[i for i in range(num_atoms)])
        logging.info(f"The FULL group " + group_summary(full_group))

        accepted_perms = []
        for perm_sage in full_group:
            perm = perm_sage.dict()
            assert self.validate_permutation(perm), f"{perm} is invalid"
            if self.check_permutation_stereo(perm):
                accepted_perms.append(perm_sage)
        short_group = PermutationGroup(accepted_perms,
                                       domain=[i for i in range(num_atoms)])
        logging.info(f"The short group " + group_summary(short_group))

        assert len(short_group) == len(accepted_perms)
        assert short_group.is_subgroup(full_group)

        return short_group

    def validate_permutation(self, perm: dict[int, int]) -> bool:
        num_atoms = self.molgraph.number_of_nodes()
        for x in itertools.chain(perm.keys(), perm.values()):
            if not (isinstance(x, int) and x >= 0 and x < num_atoms):
                return False
        return True

    def check_permutation_stereo(self, perm: dict[int, int]) -> bool:
        num_atoms = self.molgraph.number_of_nodes()
        np_reorder = np.array([perm[i] for i in range(num_atoms)])
        original_coords = self.p[0].xyz
        permuted_coords = original_coords[np_reorder]
        for stereocenter in self.stereocenters.values():
            if not stereocenter.validate_coords(
                    permuted_coords,
                    threshold=self.config.stereocenter_rmsd_threshold):
                return False
        return True

    def represent_group_on_dihedrals(
        self,
        full_group: PermutationGroup,
        dofs_list: list[tuple[int, int, int, int]],
    ):
        short_dofs = [(dof[1], dof[2]) for dof in dofs_list]
        short_dofs_neg = [(dof[2], dof[1]) for dof in dofs_list]

        def extract_dihedrals(m: ringo.MolProxy) -> np.ndarray:
            res = deg2m(m.z(*[i + 1 for i in atoms]) for atoms in dofs_list)
            return res

        def get_permutation_on_dofs(
                perm: dict[int, int]) -> tuple[np.ndarray, dict[int, int]]:
            perm_matrix = np.zeros((len(dofs_list), len(dofs_list)))
            dof_map = {}
            sign_matrs = []
            for source_i, source_atoms in enumerate(short_dofs):
                target_atoms = tuple(perm[i] for i in source_atoms)
                if target_atoms in short_dofs:
                    target_i = short_dofs.index(target_atoms)
                    sign_matrs.append(np.eye(2))
                elif target_atoms in short_dofs_neg:
                    target_i = short_dofs_neg.index(target_atoms)
                    sign_matrs.append(np.diag([1.0, -1.0]))
                else:
                    raise Exception(
                        "Rotamer group is not closed under dofs permutations!")
                perm_matrix[source_i, target_i] = 1.0
                dof_map[source_i] = target_i

            full_perm_matrix = np.kron(perm_matrix, np.eye(2))
            full_sign_matrix = scipy.linalg.block_diag(*sign_matrs)
            res_matrix = full_sign_matrix @ full_perm_matrix
            # ic(full_perm_matrix, full_sign_matrix, res_matrix)
            return res_matrix, dof_map

        start_xyz = self.p[0].xyz
        start_matr = extract_dihedrals(self.p[0])
        ic(m2deg(start_matr))

        perm_representations = []
        dof_actions = []
        for i, perm_sage in enumerate(full_group):
            perm = perm_sage.dict()
            ic(perm)
            new_xyz = start_xyz[np.array(
                [perm[i] for i in range(self.p.natoms)])]
            self.p.include_from_xyz(new_xyz, "test")
            final_matr = extract_dihedrals(self.p[1])
            ic(m2deg(final_matr))
            del self.p[1]

            dof_perm_matr, dof_perm_map = get_permutation_on_dofs(perm)
            ic(dof_perm_matr, dof_perm_map)

            part_result = dof_perm_matr @ start_matr @ np.linalg.inv(
                dof_perm_matr)
            ic(m2deg(part_result))
            rot_correction = final_matr @ np.linalg.inv(part_result)
            perm_representations.append(
                (rot_correction @ dof_perm_matr, np.linalg.inv(dof_perm_matr)))
            dof_actions.append(dof_perm_map)
        ic(perm_representations)
        ic(dof_actions)
        return perm_representations, dof_actions


class Stereocenter:

    def __init__(self,
                 full_coords: np.ndarray,
                 center_atom: int,
                 nb_atoms: list[int],
                 atom_symbol: Optional[str] = None):
        self.center_atom = center_atom
        self.atom_symbol = atom_symbol
        self.nb_order = sorted(nb_atoms)
        self.localcoords = self.fullcoords_to_local(full_coords)
        self.coords_matrix = self.localcoords_to_matrix(self.localcoords)

    def __repr__(self):
        return (
            f"Stereocenter at atom {self.atom_symbol}{self.center_atom+1}, "
            f"fullrank={self.is_full_rank()}, num_nbs={len(self.nb_order)}")

    def fullcoords_to_local(self,
                            full_coords: np.ndarray) -> dict[int, np.ndarray]:
        center_xyz = full_coords[self.center_atom, :]
        nb_xyzs = {i: full_coords[i, :] for i in self.nb_order}
        result = {k: v - center_xyz for k, v in nb_xyzs.items()}
        result = {k: v / np.linalg.norm(v) for k, v in result.items()}
        return result

    def localcoords_to_matrix(
            self, localcoords: dict[int, np.ndarray]) -> np.ndarray:
        result = np.array([localcoords[i] for i in self.nb_order])
        return result

    def is_full_rank(self) -> bool:
        """This potentially can be use to ignore some simple stereocenters"""
        rank, is_full = determine_span(self.coords_matrix)
        return is_full

    def validate_coords(self, full_coords: np.ndarray,
                        threshold: float) -> bool:
        check_localmatrix = self.localcoords_to_matrix(
            self.fullcoords_to_local(full_coords))
        rmsd_with_inversion = kabsch_rmsd(self.coords_matrix,
                                          check_localmatrix,
                                          allow_inversion=True)
        rmsd_without_inversion = kabsch_rmsd(self.coords_matrix,
                                             check_localmatrix,
                                             allow_inversion=False)
        assert rmsd_with_inversion < threshold, f"rmsd_with_inversion={rmsd_with_inversion}"
        return rmsd_without_inversion < threshold
