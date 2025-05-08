import ringo
import networkx as nx
import numpy as np

from .rotamer import RotamerAnalyzer
from .config import InitialGeneratorConfig

import time
import logging


# TODO: These two functions are trash. Rewrite based on general symmetry.
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


def run_grid_search(
    mol: ringo.Molecule,
    dihedral_values: list[list[float]],
    config: InitialGeneratorConfig,
    input_sdf: str,
) -> ringo.Confpool:
    # TODO: Shrink grid based on symmetry
    p = ringo.Confpool()
    dofs_list, dofs_values = mol.get_ps()

    # rotanalyzer = RotamerAnalyzer.from_sdf(input_sdf, config.rotamer_analysis)
    # rotamer_group = rotanalyzer.get_rotamer_group()
    # logging.info(f"Got this rotamer group: {rotamer_group}")
    # rotanalyzer.represent_group_on_dihedrals(rotamer_group, dofs_list)

    graph = mol.molgraph_access()
    for n, sym in enumerate(mol.get_symbols()):
        graph.nodes[n]['symbol'] = sym
    custom_preferences = {}

    num_conn_comp = sum(1 for i in nx.connected_components(graph))
    assert num_conn_comp == 1

    # for i, (atoms, value) in enumerate(zip(dofs_list, dofs_values)):
    #     if is_boring_dihedral(graph, atoms[1], atoms[2]):
    #         custom_preferences[i] = [value]

    preferences = {
        #MIWTER
        # (3, 4, 5, 7): [180.0],
        # (15, 16, 17, 19): [180.0],
        # (26, 27, 1, 2): [180.0],
        # (15, 38, 39, 40): [90.0],
        # 2IYA
        # (3, 4, 14, 5): [180.0],
        # (37, 38, 42, 60): [60.0, 180.0],
        # (36, 37, 41, 44): [-60.0, 180.0],
        # (25, 26, 33, 70): [-60.0, 180.0],
        # (25, 26, 33, 70): [-60.0, 180.0],
        # (26, 27, 31, 34): [-60.0, 180.0],
    }

    for i, (atoms, value) in enumerate(zip(dofs_list, dofs_values)):
        if atoms in preferences:
            custom_preferences[i] = preferences[atoms]
        elif is_boring_dihedral(graph, atoms[1], atoms[2]):
            ic('boring', atoms)
            custom_preferences[i] = [dihedral_values[i][0]]
        else:
            custom_preferences[i] = dihedral_values[i]

    logging.info(f"Using dihedral preferences:")
    for i, values in custom_preferences.items():
        logging.info(f"{i}) {dofs_list[i]} - {values}")

    cs_settings = ringo.SysConfsearchInput(
        molecule=mol,
        pool=p,
        rmsd_settings='default',
        default_dihedrals=[-60.0, 60.0, 180.0],
        custom_preferences=custom_preferences,
        clear_feed=False,
        show_status=config.show_status,
        nthreads=1,
    )

    start_time = time.time()
    ringo.systematic_sampling(cs_settings)
    finish_time = time.time()

    logging.info(f"Grid search time spent = {(finish_time-start_time):.2f}")
    return p
