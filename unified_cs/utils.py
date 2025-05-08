from rdkit import Chem
from rdkit.Chem import AllChem

import networkx as nx
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import ringo
import vf3py

import tempfile
import os
from contextlib import contextmanager
import logging
import copy
from typing import TypeVar, Callable, Optional, Generator

from pydantic.dataclasses import dataclass
from pydantic import ConfigDict

pyd_dataclass = dataclass(config=ConfigDict(arbitrary_types_allowed=True))


def rdmol_to_graph(mol: Chem.Mol) -> nx.Graph:
    graph = nx.Graph()

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        symbol = atom.GetSymbol()
        charge = atom.GetFormalCharge()
        graph.add_node(idx, symbol=symbol, charge=charge)

    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()

        bond_order = {
            Chem.rdchem.BondType.SINGLE: 1.0,
            Chem.rdchem.BondType.DOUBLE: 2.0,
            Chem.rdchem.BondType.TRIPLE: 3.0,
            Chem.rdchem.BondType.AROMATIC: 1.5,
        }.get(bond_type, None)

        if bond_order is None:
            raise ValueError(f"Unsupported bond type: {bond_type}")

        graph.add_edge(begin, end, bondorder=bond_order)

    for node, data in graph.nodes(data=True):
        order_sum = sum(graph[node][nb_node]['bondorder']
                        for nb_node in graph.neighbors(node))
        # assert order_sum == int(order_sum), (
        #     f"Fractional valence detected for {data['symbol']}{node+1}")
        data['valence'] = order_sum

    return graph


def graph_to_rdmol(graph: nx.Graph) -> Chem.Mol:
    mol = Chem.RWMol()
    idx_map = {}

    sorted_nodes = sorted([i for i in graph.nodes])
    for node in sorted_nodes:
        data = graph.nodes[node]
        symbol = data.get('symbol')
        assert symbol is not None, f"Node {node} is missing 'symbol' attribute."

        charge = data.get('charge', None)
        assert charge is not None, f"Atom charge is missing for {node}"

        atom = Chem.Atom(symbol)
        atom.SetFormalCharge(charge)
        idx = mol.AddAtom(atom)
        idx_map[node] = idx

    for u, v, data in graph.edges(data=True):
        bondorder = data.get('bondorder')
        assert bondorder is not None, f"Edge ({u}, {v}) is missing 'bondorder' attribute."

        bond_type = {
            1.0: Chem.rdchem.BondType.SINGLE,
            2.0: Chem.rdchem.BondType.DOUBLE,
            3.0: Chem.rdchem.BondType.TRIPLE,
            1.5: Chem.rdchem.BondType.AROMATIC
        }.get(bondorder, None)
        assert bond_type is not None, f"Unsupported bond order: {bondorder}"

        mol.AddBond(idx_map[u], idx_map[v], bond_type)

    return mol.GetMol()


def edge_neighbors(G, edge, *, data: bool = False):
    """
    Generator that yields edges which are neighbors of a given edge.
    Two edges are considered neighbors if they share at least one common node.
    
    Parameters:
        G (networkx.Graph or networkx.DiGraph): The input graph.
        edge (tuple): A tuple representing the edge (u, v).
        data (bool): If True, yields (u, v, data_dict). Else yields (u, v).
    
    Yields:
        tuple: An edge neighboring the given edge.
    """
    u, v = edge

    def canonical(e):
        return e if G.is_directed() else tuple(sorted(e))

    original = canonical(edge)
    seen = set()

    for node in (u, v):
        for neighbor in G.neighbors(node):
            candidate_edge = (node, neighbor)
            candidate = canonical(candidate_edge)
            if candidate == original or candidate in seen:
                continue
            seen.add(candidate)
            if data:
                edge_data = G.get_edge_data(*candidate_edge)
                yield (*candidate_edge, edge_data)
            else:
                yield candidate


def build_model_dihedral(
    molgraph: nx.Graph,
    dihedral: tuple[int, int, int, int],
) -> nx.Graph:
    if not all(
            molgraph.has_edge(dihedral[i], dihedral[j])
            for i, j in ((0, 1), (1, 2), (2, 3))):
        return None

    selected_atoms = [dihedral[1], dihedral[2]]

    def add_nbs(atom: int):
        for nb in molgraph.neighbors(atom):
            if nb not in selected_atoms:
                selected_atoms.append(nb)

    add_nbs(dihedral[1])
    add_nbs(dihedral[2])

    bond_subgraph: nx.Graph = molgraph.subgraph(selected_atoms).copy()

    # graph_correction_methods: list[Callable[[nx.Graph], None]] = [
    #     lone_aromatic_to_double_correction,
    #     pair_aromatic_to_singledouble_correction,
    # ]
    # ic('Before', bond_subgraph.edges(data=True))
    # for m in graph_correction_methods:
    #     m(bond_subgraph)
    # ic('After', bond_subgraph.edges(data=True))

    for node, node_data in bond_subgraph.nodes(data=True):
        node_data['extra'] = False

    new_nodes = []
    new_edges = []
    total_h_count = 0
    for node, node_data in bond_subgraph.nodes(data=True):
        logging.debug(f"New node {node}")
        current_valence = sum(bond_subgraph[node][nb]['bondorder']
                              for nb in bond_subgraph.neighbors(node))
        assert current_valence == int(current_valence), (
            f"Fractional valence detected for {node_data['symbol']}{node+1}")
        current_valence = int(current_valence)
        # current_valence = len(list(nb for nb in bond_subgraph.neighbors(node)))

        assert node_data['valence'] == int(node_data['valence']), (
            f"Fractional valence detected for {node_data['symbol']}{node+1}")
        node_valence = int(node_data['valence'])
        assert current_valence <= node_valence
        num_hydrogens = node_valence - current_valence

        for new_h_index in range(num_hydrogens):
            h_label = f"H_{total_h_count}"
            new_nodes.append((h_label, {
                'symbol': 'H',
                'valence': 1.0,
                'charge': 0,
                'extra': True,
            }))
            new_edges.append(((h_label, node), {'bondorder': 1.0}))
            total_h_count += 1

    for label, attrs in new_nodes:
        bond_subgraph.add_node(label, **attrs)
    for labels, attrs in new_edges:
        bond_subgraph.add_edge(*labels, **attrs)

    for a, b, data in bond_subgraph.edges(data=True):
        data['scanned'] = False
    bond_subgraph[dihedral[1]][dihedral[2]]['scanned'] = True

    ic(bond_subgraph.nodes(data=True))
    ic(bond_subgraph.edges(data=True))
    return bond_subgraph


T = TypeVar('T')


def get_equivalence_classes(
    domain: list[T] | dict[int, T],
    eq: Callable[[T, T], bool],
) -> tuple[list[int], dict[int, int]]:
    reps = []
    element_to_class = {}
    if isinstance(domain, list):
        for i, x in enumerate(domain):
            for rep_idx, r in enumerate(reps):
                if eq(x, domain[r]):
                    element_to_class[i] = rep_idx
                    break
            else:
                reps.append(i)
                element_to_class[i] = len(reps) - 1
    elif isinstance(domain, dict):
        for i, x in domain.items():
            for rep_idx, r in enumerate(reps):
                if eq(x, domain[r]):
                    element_to_class[i] = rep_idx
                    break
            else:
                reps.append(i)
                element_to_class[i] = len(reps) - 1

    return reps, element_to_class


def wrap_angle(angle):
    """Wraps any angle (in radians) to the range [0.0, 2π)."""
    return angle % (2 * np.pi)


def optimize_rdmol(
    mol: Chem.Mol,
    xyzs: np.ndarray,
    dihedral_constraints=[],
) -> tuple[np.ndarray, float]:
    mol.RemoveAllConformers()
    conf = Chem.Conformer(mol.GetNumAtoms())

    for atom_idx in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(atom_idx, xyzs[atom_idx])
    mol.AddConformer(conf)

    mp = AllChem.MMFFGetMoleculeProperties(mol)
    ff = AllChem.MMFFGetMoleculeForceField(mol, mp)
    if len(dihedral_constraints) > 0:
        for atoms, value in dihedral_constraints:
            ff.MMFFAddTorsionConstraint(
                *atoms,
                False,
                # Minus is due to different sign convetions
                float(-value * ringo.RAD2DEG),
                float(-value * ringo.RAD2DEG),
                1e4,
            )

    try_i = 0
    return_code = 1
    while try_i < 10 and return_code != 0:
        return_code = ff.Minimize(maxIts=10000)
        try_i += 1
    assert return_code == 0

    energy = ff.CalcEnergy()
    opt_xyzs = get_xyzs_from_rdmol(mol)
    return opt_xyzs, energy


def calc_rdmol_singlepoint(mol: Chem.Mol) -> float:
    mp = AllChem.MMFFGetMoleculeProperties(mol)
    ff = AllChem.MMFFGetMoleculeForceField(mol, mp)
    energy = ff.CalcEnergy()
    return energy


def get_xyzs_from_rdmol(mol: Chem.Mol) -> np.ndarray:
    opt_xyzs = np.zeros((mol.GetNumAtoms(), 3))
    for i in range(mol.GetNumAtoms()):
        pos = mol.GetConformer().GetAtomPosition(i)
        opt_xyzs[i, 0] = pos.x
        opt_xyzs[i, 1] = pos.y
        opt_xyzs[i, 2] = pos.z
    return opt_xyzs


def compute_graph_checksum(
    graph: nx.Graph,
    node_attrs: list[str],
    edge_attrs: list[str],
) -> str:
    """
    Computes a Weisfeiler-Lehman hash of a graph that is invariant under 
    isomorphisms preserving the specified node and edge attributes.

    Parameters:
        graph (nx.Graph): The input graph.
        node_attrs (list[str]): List of node attribute keys to include in the hash.
        edge_attrs (list[str]): List of edge attribute keys to include in the hash.

    Returns:
        str: The isomorphism-invariant checksum of the graph.
    """
    for node, data in graph.nodes(data=True):
        data['_total'] = tuple([data[key] for key in node_attrs])
    for a, b, data in graph.edges(data=True):
        data['_total'] = tuple([data[key] for key in edge_attrs])
    hash = weisfeiler_lehman_graph_hash(
        graph,
        node_attr='_total',
        edge_attr='_total',
    )
    return hash


def same_graphs(first_graph: nx.Graph, second_graph: nx.Graph,
                node_attrs: list[str], edge_attrs: list[str]) -> bool:
    """
    Checks if two graphs are isomorphic with respect to specified node and edge attributes.

    Parameters:
        first_graph (nx.Graph): The first graph.
        second_graph (nx.Graph): The second graph.
        node_attrs (list[str]): List of node attribute keys to compare.
        edge_attrs (list[str]): List of edge attribute keys to compare.

    Returns:
        bool: True if the graphs are isomorphic considering the given attributes, False otherwise.
    """
    for graph in (first_graph, second_graph):
        for node, data in graph.nodes(data=True):
            data['_total'] = tuple([data[key] for key in node_attrs])
        for a, b, data in graph.edges(data=True):
            data['_total'] = tuple([data[key] for key in edge_attrs])

    return vf3py.are_isomorphic(
        first_graph,
        second_graph,
        node_match=lambda a, b: a['_total'] == b['_total'],
        edge_match=lambda a, b: a['_total'] == b['_total'],
    )


def get_unique_dihedrals(graph: nx.Graph) -> list[tuple[int, int, int, int]]:
    pattern = nx.path_graph(4)
    matches = [
        tuple(match[i] for i in range(4))
        for match in vf3py.get_subgraph_isomorphisms(subgraph=pattern,
                                                     graph=graph)
    ]
    reps_indices, _ = get_equivalence_classes(
        matches,
        eq=lambda a, b:
        (a[1] == b[1] and a[2] == b[2]) or (a[1] == b[2] and a[1] == b[2]))
    reps = [matches[i] for i in reps_indices]
    return reps


def rdmol_to_coord_matrix(mol: Chem.Mol, conformer_index=-1) -> np.ndarray:
    """
    Converts an RDKit molecule into a NumPy coordinate matrix.
    
    Parameters:
        mol (Chem.Mol): RDKit molecule with at least one conformer.

    Returns:
        np.ndarray: Coordinate matrix of shape (num_atoms, 3)
    """
    assert mol.GetNumConformers() > 0, (
        "Molecule has no conformers with 3D coordinates.")

    conf = mol.GetConformer(conformer_index)
    num_atoms = mol.GetNumAtoms()
    coords = np.zeros((num_atoms, 3), dtype=np.float32)

    for i in range(num_atoms):
        pos = conf.GetAtomPosition(i)
        coords[i] = [pos.x, pos.y, pos.z]

    return coords


def gen_labels(values: list[float],
               threshold: float,
               period: Optional[float] = None
               ) -> tuple[list[int], dict[int, list[float]]]:
    """
    Cluster values with given threshold. If a period is provided, values are treated as circular,
    so that, e.g., x and x+period are considered equal.

    Args:
        values: List of floats to cluster.
        threshold: Distance threshold for stopping cluster merging.
        period: (Optional) Defines a period for the input values. If provided, the distance between
                any two values x and y is defined as: 
                d(x, y) = min(|x - y|, period - |x - y|).

    Returns:
        A tuple containing:
            - cluster_labels: a list of integer labels corresponding to each input value.
            - clusters: a dictionary mapping each cluster label to the list of original values in that cluster.
    """
    if len(values) == 0:
        return [], {}
    if len(values) == 1:
        return [0], {0: values}

    values_arr = np.array(values)
    if period is not None:
        values_mod = values_arr % period
        diffs = np.abs(values_mod.reshape(-1, 1) - values_mod.reshape(1, -1))
        dist_matrix = np.minimum(diffs, period - diffs)

        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric='precomputed',
            linkage='single',
            distance_threshold=threshold,
        ).fit(dist_matrix)
    else:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            linkage='single',
            distance_threshold=threshold,
        ).fit(values_arr.reshape(-1, 1))

    value_clusters: dict[int, list[float]] = {}
    cluster_labels = clustering.labels_.tolist()
    for label, value in zip(cluster_labels, values):
        value_clusters.setdefault(label, []).append(value)

    return cluster_labels, value_clusters


def determine_span(M: np.ndarray, threshold=1e-3) -> tuple[int, bool]:
    """
    Determines if the rows of matrix M span R^3 or only a subspace.
    
    Parameters:
    - M: A numpy array of shape (n, 3)
    - threshold: A fraction (relative to the largest singular value) 
      below which singular values are considered zero.
    
    Returns:
    - effective_rank: The effective rank of M after thresholding.
    - is_full_span: Boolean indicating whether the rows span R^3.
    """

    assert M.shape[1] == 3
    u, s, vt = np.linalg.svd(M)
    max_singular = s[0]
    effective_rank = np.sum(s > threshold * max_singular)
    is_full_span = (effective_rank == 3)
    return effective_rank, is_full_span


def kabsch_rmsd(p_all: np.ndarray,
                q_all: np.ndarray,
                skip_centering: bool = True,
                allow_inversion: bool = False) -> float:
    p_coord = np.array(copy.deepcopy(p_all))
    q_coord = np.array(copy.deepcopy(q_all))

    if not skip_centering:
        p_coord -= p_coord.mean(axis=1)
        q_coord -= q_coord.mean(axis=1)

    C = np.dot(p_coord.T, q_coord)
    V, _, W = np.linalg.svd(C)

    if not allow_inversion and (np.linalg.det(V) * np.linalg.det(W) < 0.0):
        V[:, 2] = -V[:, 2]

    U = np.dot(V, W)
    Udet = np.linalg.det(U)
    assert np.isclose(Udet, 1.0) or np.isclose(Udet, -1.0), f"Udet = {Udet}"

    p_coord = np.dot(p_coord, U)
    diff = p_coord - q_coord
    num_atoms = p_coord.shape[0]
    res = np.sqrt((diff * diff).sum() / num_atoms)
    return res


def group_summary(group) -> str:
    generators = group.gens_small()
    return (f"has {len(group)} elements and {len(generators)} generators:\n" +
            "\n".join(f"{i}. {g.cycle_string()}"
                      for i, g in enumerate(generators, start=1)))


def boring_check_oneside(g: nx.Graph, full_graph: nx.Graph, node: int) -> bool:
    neighbors = [n for n in g.neighbors(node)]
    some_symbol = full_graph.nodes[neighbors[0]]['symbol']
    return (g.number_of_nodes() == len(neighbors) + 1
            and g.number_of_edges() == len(neighbors) and len(neighbors) == 3
            and all(full_graph.nodes[n]['symbol'] == some_symbol
                    for n in neighbors))


def is_boring_dihedral(graph: nx.Graph, central_atom: int,
                       connected_atom: int) -> bool:
    g = nx.Graph()
    g.add_edges_from(graph.edges)
    g.remove_edge(central_atom, connected_atom)
    cc = [c for c in nx.connected_components(g)]
    if len(cc) == 1:
        return False

    assert len(cc) == 2
    first_cc, second_cc = cc[0], cc[1]
    first_subg: nx.Graph = g.subgraph(first_cc)
    second_subg: nx.Graph = g.subgraph(second_cc)
    if first_subg.has_node(central_atom):
        return (boring_check_oneside(first_subg, graph, central_atom)
                or boring_check_oneside(second_subg, graph, connected_atom))
    else:
        return (boring_check_oneside(first_subg, graph, connected_atom)
                or boring_check_oneside(second_subg, graph, central_atom))


def get_boring_dihedrals(graph: nx.Graph) -> list[tuple[int, int]]:
    if len(list(nx.connected_components(graph))) > 1:
        result = []
        for comp in nx.connected_components(graph):
            result += get_boring_dihedrals(graph.subgraph(comp))
        return result

    boring_dihedrals = []
    for u, v in graph.edges:
        # Check if each side of the bond has at least one neighbor (i.e., it's part of a dihedral)
        if len(list(graph.neighbors(u))) > 1 and len(list(
                graph.neighbors(v))) > 1:
            if is_boring_dihedral(graph, u, v):
                boring_dihedrals.append((u, v))
            elif is_boring_dihedral(graph, v, u):
                boring_dihedrals.append((v, u))
    return boring_dihedrals


@contextmanager
def temp_config_copy(original_path: str,
                     subs: dict[str, str]) -> Generator[str, None, None]:
    with tempfile.NamedTemporaryFile(mode='w+', delete=False,
                                     suffix='.yaml') as temp:
        with open(original_path, 'r') as f:
            content: str = f.read()

        new_content: str = content.format(**subs)
        temp.write(new_content)
        temp.flush()

        yield temp.name

    # Optionally, you can delete the tempfile here if desired
    os.remove(temp.name)


# These are not necessary when doing kekulization in RDKit
# def lone_aromatic_to_double_correction(graph: nx.Graph) -> None:
#     lone_edges = []
#     is_aromatic = lambda data: data['bondorder'] == 1.5
#     for edge_a, edge_b, edge_data in graph.edges(data=True):
#         edge = (edge_a, edge_b)
#         if is_aromatic(edge_data) and all(
#                 not is_aromatic(nbedge_data)
#                 for _, __, nbedge_data in edge_neighbors(
#                     graph, edge, data=True)):
#             lone_edges.append(edge)

#     for edge in lone_edges:
#         graph[edge[0]][edge[1]]['bondorder'] = 2.0

# def pair_aromatic_to_singledouble_correction(graph: nx.Graph) -> None:
#     aromatic_pairs = []
#     is_aromatic = lambda data: data['bondorder'] == 1.5

#     for edge_a, edge_b, edge_data in graph.edges(data=True):
#         if not is_aromatic(edge_data):
#             continue

#         edge = tuple(sorted((edge_a, edge_b)))
#         aromatic_nbedges = [(nbedge_a, nbedge_b)
#                             for nbedge_a, nbedge_b, nbedge_data in
#                             edge_neighbors(graph, edge, data=True)
#                             if is_aromatic(nbedge_data)]
#         if len(aromatic_nbedges) == 1:
#             aromatic_pairs.append(sorted(tuple((edge, aromatic_nbedges[0]))))

#     for first_edge, second_edge in set(aromatic_pairs):
#         graph[first_edge[0]][first_edge[1]]['bondorder'] = 2.0
#         graph[second_edge[0]][second_edge[1]]['bondorder'] = 1.0
