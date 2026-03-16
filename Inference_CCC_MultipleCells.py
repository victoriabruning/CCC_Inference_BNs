
"""
Cell-Cell Communication Learning Module
========================================
This module provides functions to learn Boolean network models for cell-cell communication
through receptors and ligands using the BoNesis framework.

Supports any number of cells, ligands, and receptors.
"""

import bonesis # type: ignore
from colomoto import minibn # type: ignore
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings


# ==================== DATA SETUP ====================



def create_observation_data(
    R: Optional[Dict[str, Dict]] = None,
    L: Optional[Dict[str, Dict]] = None,
    R_cells: Optional[List[str]] = None, # NEW: which cells to include receptors for
    L_cells: Optional[List[str]] = None,  # NEW: which cells to include ligands for
):
    all_L_cells = list(L.keys())
    all_R_cells = list(R.keys())
    R_cells = R_cells or all_R_cells
    L_cells = L_cells or all_L_cells  # default: all cells in L
    timepoints = list(next(iter(L.values())).keys())

    # Merge observations only for selected cells
    obs = {}
    for T in timepoints:
        merged = {}
        for c in L_cells:
            merged.update(L[c][T])
        for c in R_cells:
            merged.update(R[c][T])
        obs[T] = merged

    obs_df = pd.DataFrame.from_dict(obs, orient="index")

    all_receptors_list = [n for n in obs_df.columns if n[0] == "R"]
    all_ligands_list = [n for n in obs_df.columns if n[0] == "L"]

    # Compressed observations
    R_up = obs_df[all_receptors_list].iloc[0::2]
    R_up.index = obs_df.index[:R_up.shape[0]]
    L_up = obs_df[all_ligands_list].iloc[1::2]
    L_up.index = obs_df.index[1:L_up.shape[0] + 1]
    compr_obs_df = pd.concat([R_up, L_up], axis=1).astype("Int8")

    # Build cells structure: only selected cells
    active_cells = list(dict.fromkeys(R_cells + L_cells))  # preserve order, no duplicates
    cells = {}
    for c in active_cells:
        cells[c] = {
            "R": list(next(iter(R[c].values())).keys()) if c in R_cells else [],
            "L": list(next(iter(L[c].values())).keys()) if c in L_cells else [],
        }

    all_ligands = [node for c in L_cells for node in cells[c]["L"]]
    all_receptors = [node for c in R_cells for node in cells[c]["R"]]

    # Define influences
    influences = []
    # Any selected ligand can influence receptors of R_cells
    influences += [
        (lig, rec, {"sign": 0})
        for c in R_cells for rec in cells[c]["R"]
        for d in L_cells for lig in cells[d]["L"]
    ]
    # Within R_cells: link receptors to ligands of the same cell (only if also in L_cells)
    influences += [
        (rec, lig, {"sign": 0})
        for c in R_cells for rec in cells[c]["R"]
        for lig in cells[c]["L"] if c in L_cells
    ]

    return obs_df, compr_obs_df, cells, influences, all_receptors, all_ligands




# ==================== SYNTHESIS FUNCTIONS ====================



def receptors_synthesis(compr_obs_df: pd.DataFrame, cells: Dict, influences: List):
    """
    Synthesize Boolean functions for receptors taking into account all ligands (of all cells)
    
    Args:
        compr_obs_df: Compressed observation dataframe
        cells: Cell structure dictionary
        influences: List of influence tuples
        
    Returns:
        tuple: (bo, nodes) - BoNesis object and list of nodes
    """
    R = []
    L = []
    for cell_data in cells.values():
        R += list(cell_data["R"])
        L += list(cell_data["L"])
    nodes = R + L
    
    subnet = [(n, m, s) for (n, m, s) in influences if m in nodes]
    
    dom = bonesis.InfluenceGraph(
        subnet, 
        maxclause=10, 
        exact=False,
        allow_skipping_nodes=False,
        canonic=True
    )
    
    compr_obs = compr_obs_df.to_dict(orient="index")
    bo = bonesis.BoNesis(dom, compr_obs)
        
    x1 = ~bo.obs("T1")
    x2 = ~bo.obs("T2")
    x3 = ~bo.obs("T3")
    x4 = ~bo.obs("T4")
    x5 = ~bo.obs("T5")
    x6 = ~bo.obs("T6")

    x1 >= x2 >= x3 >= x4 >= x5 >= x6
    
    # LEARN RECEPTORS
    with bo.mutant(x2[L]): 
        bo.fixed(x2)
    with bo.mutant(x3[L]):
        bo.fixed(x3)
    with bo.mutant(x4[L]):
        bo.fixed(x4)
    with bo.mutant(x5[L]):
        bo.fixed(x5)
    with bo.mutant(x6[L]):
         bo.fixed(x6)

    return bo, nodes



def ligands_synthesis(compr_obs_df: pd.DataFrame, cells: Dict, influences: List):
    """
    Synthesize Boolean functions for ligands for each cell, taking into account only its receptors
    
    Args:
        compr_obs_df: Compressed observation dataframe
        cells: Cell structure dictionary
        influences: List of influence tuples
        
    Returns:
        tuple: (bo, nodes) - BoNesis object and list of nodes
    """
    R = []
    L = []
    for cell_data in cells.values():
        R += list(cell_data["R"])
        L += list(cell_data["L"])
    nodes = R + L
    
    subnet = [(n, m, s) for (n, m, s) in influences if m in nodes]
    
    dom = bonesis.InfluenceGraph(
        subnet, 
        maxclause=10, 
        exact=False,
        allow_skipping_nodes=False,
        canonic=True
    )
    
    compr_obs = compr_obs_df.to_dict(orient="index")
    bo = bonesis.BoNesis(dom, compr_obs)
        
    x1 = ~bo.obs("T1")
    x2 = ~bo.obs("T2")
    x3 = ~bo.obs("T3")
    x4 = ~bo.obs("T4")
    x5 = ~bo.obs("T5")
    #x6 = ~bo.obs("T6")

    x1 >= x2 >= x3 >= x4 >= x5 #>= x6
    

    # LEARN LIGANDS
    with bo.mutant(x1[R]):
        bo.fixed(x2[L])
    with bo.mutant(x2[R]):
        bo.fixed(x3[L])
    with bo.mutant(x3[R]):
        bo.fixed(x4[L])
    with bo.mutant(x4[R]):
         bo.fixed(x5[L])
    # with bo.mutant(x5[R]):
    #     bo.fixed(x6[L])

    return bo, nodes




def all_cells_synthesis(compr_obs_df: pd.DataFrame, cells: Dict, influences: List):
    """
    Synthesize Boolean functions for receptors AND ligands of BOTH cells simultaneously.
    
    Args:
        compr_obs_df: Compressed observation dataframe
        cells: Cell structure dictionary
        influences: List of influence tuples
        
    Returns:
        tuple: (bo, nodes) - BoNesis object and list of nodes
    """
    R = []
    L = []
    for cell_data in cells.values():
        R += list(cell_data["R"])
        L += list(cell_data["L"])
    nodes = R + L
    
    subnet = [(n, m, s) for (n, m, s) in influences if m in nodes]
    
    dom = bonesis.InfluenceGraph(
        subnet, 
        maxclause=10, 
        exact=False,
        allow_skipping_nodes=False,
        canonic=True
    )
    
    compr_obs = compr_obs_df.to_dict(orient="index")
    bo = bonesis.BoNesis(dom, compr_obs)
        
    x1 = ~bo.obs("T1")
    x2 = ~bo.obs("T2")
    x3 = ~bo.obs("T3")
    x4 = ~bo.obs("T4")


    x1 >= x2 >= x3 >= x4 
    
    # LEARN RECEPTORS
    with bo.mutant(x2[L]): 
        bo.fixed(x2)
    with bo.mutant(x3[L]):
        bo.fixed(x3)
    with bo.mutant(x4[L]):
        bo.fixed(x4)


    # LEARN LIGANDS
    with bo.mutant(x1[R]):
        bo.fixed(x2[L])
    with bo.mutant(x2[R]):
        bo.fixed(x3[L])
    with bo.mutant(x3[R]):
        bo.fixed(x4[L])

    return bo, nodes




# ====================== PLOT INFLUENCE GRAPHS ====================



def compute_node_positions(all_nodes, x_spacing=200, y_spacing=200, x_margin=150, y_margin=100):
    """
    Compute (x, y) positions for nodes based on their names.

    Node name format: {R|L}_{celltype}_{nodename}
    - Cell types are laid out left to right, ordered by first appearance
    - R nodes are placed on the top row, L nodes on the bottom row
    - Within each cell type, nodes are sorted alphabetically and spaced evenly

    Parameters
    ----------
    all_nodes : iterable of str
        All node names present in the graphs.
    x_spacing : int
        Horizontal pixel spacing between nodes of the same cell type.
    y_spacing : int
        Vertical pixel spacing between the R row and the L row.
    x_margin : int
        Left margin in pixels.
    y_margin : int
        Top margin in pixels.

    Returns
    -------
    positions : dict
        Mapping node_name -> (x, y).
    cell_centers : dict
        Mapping celltype -> x center coordinate (useful for annotations).
    """
    from collections import defaultdict

    receptors = defaultdict(list)  # celltype -> [node_name, ...]
    ligands   = defaultdict(list)
    cell_order = []

    for node in sorted(all_nodes):
        parts = node.split("_", 2)          # ["R", "CLL", "nodename"]
        if len(parts) < 3:
            continue
        kind, celltype, _ = parts
        if celltype not in cell_order:
            cell_order.append(celltype)
        if kind == "R":
            receptors[celltype].append(node)
        elif kind == "L":
            ligands[celltype].append(node)

    # Sort nodes within each cell type for consistent ordering
    for c in cell_order:
        receptors[c].sort()
        ligands[c].sort()

    # Determine the maximum number of nodes per row across all cell types
    # so that cell-type columns are evenly spaced
    max_r = max((len(receptors[c]) for c in cell_order), default=1)
    max_l = max((len(ligands[c])   for c in cell_order), default=1)
    col_width = max(max_r, max_l) * x_spacing   # width allocated per cell type

    positions   = {}
    cell_centers = {}

    for ci, celltype in enumerate(cell_order):
        x_base = x_margin + ci * col_width
        cell_centers[celltype] = x_base + col_width / 2

        # --- Receptors (top row) ---
        r_nodes = receptors[celltype]
        n_r = len(r_nodes)
        for ri, node in enumerate(r_nodes):
            # Centre the group within the column
            x = x_base + (col_width - (n_r - 1) * x_spacing) / 2 + ri * x_spacing
            y = y_margin
            positions[node] = (x, y)

        # --- Ligands (bottom row) ---
        l_nodes = ligands[celltype]
        n_l = len(l_nodes)
        for li, node in enumerate(l_nodes):
            x = x_base + (col_width - (n_l - 1) * x_spacing) / 2 + li * x_spacing
            y = y_margin + y_spacing
            positions[node] = (x, y)

    return positions, cell_centers





def plot_influence_graphs(igs, title="Influence Graph",
                               x_spacing=200, y_spacing=200,
                               x_margin=150, y_margin=100):
    import ipycytoscape
    from collections import Counter

    # ------------------------------------------------------------------ #
    # 1. Collect edges and compute frequencies
    # ------------------------------------------------------------------ #
    all_edges = []
    for ig in igs:
        for source, target, data in ig.edges(data=True):
            all_edges.append((source, target, data.get('sign', 0)))

    edge_counts   = Counter(all_edges)
    max_edge_freq = max(edge_counts.values()) if edge_counts else 1

    all_nodes = set()
    edges_with_freq = {}
    for ig in igs:
        all_nodes.update(ig.nodes())
        for source, target, data in ig.edges(data=True):
            key = (source, target, data.get('sign', 0))
            edges_with_freq[key] = edge_counts[key]

    # ------------------------------------------------------------------ #
    # 2. Compute positions
    # ------------------------------------------------------------------ #
    positions, cell_centers = compute_node_positions(
        all_nodes,
        x_spacing=x_spacing,
        y_spacing=y_spacing,
        x_margin=x_margin,
        y_margin=y_margin,
    )

    # ------------------------------------------------------------------ #
    # 3. Build the Cytoscape widget
    # ------------------------------------------------------------------ #
    cytoscapeobj = ipycytoscape.CytoscapeWidget()
    cytoscapeobj.graph.directed = True

    # -- Nodes --
    for node in all_nodes:
        x, y = positions.get(node, (0, 0))

        parts = node.split("_", 2)
        if len(parts) == 3:
            kind, celltype, nodename = parts
            label = f"{kind}_{nodename}\n{celltype}"
        else:
            label = node

        cytoscapeobj.graph.add_node(
            ipycytoscape.Node(
                data={'id': node, 'label': label},
                position={'x': x, 'y': y},
                locked=True,
            )
        )

    # -- Edges --
    for (source, target, sign), freq in edges_with_freq.items():
        normalized = freq / max_edge_freq

        if sign == 1:   # activation  →  powder blue gradient
            r = int(0xA0 + (0x6A - 0xA0) * normalized)
            g = int(0xC0 + (0x9F - 0xC0) * normalized)
            b = int(0xD8 + (0xC0 - 0xD8) * normalized)
            edge_class = 'positive'
        else:            # inhibition  →  peach amber gradient
            r = int(0xF0 + (0xE8 - 0xF0) * normalized)
            g = int(0xC0 + (0xA5 - 0xC0) * normalized)
            b = int(0x90 + (0x6A - 0x90) * normalized)
            edge_class = 'negative'

        edge_color = f'#{r:02x}{g:02x}{b:02x}'

        cytoscapeobj.graph.add_edge(
            ipycytoscape.Edge(
                data={
                    'source':    source,
                    'target':    target,
                    'color':     edge_color,
                    'frequency': freq,
                    'label':     str(freq),
                },
                classes=edge_class,
            )
        )

    # ------------------------------------------------------------------ #
    # 4. Stylesheet
    # ------------------------------------------------------------------ #
    cytoscapeobj.set_style([
        {
            'selector': 'node',
            'style': {
                'background-color':    '#D3D3D3',
                'label':               'data(label)',
                'text-valign':         'center',
                'text-halign':         'center',
                'text-wrap':           'wrap',
                'color':               'black',
                'text-outline-width':  1,
                'text-outline-color':  'white',
                'font-size':           '11px',
                'font-weight':         'bold',
                'width':               50,
                'height':              50,
                'white-space':         'pre',
            }
        },
        {
            'selector': 'edge',
            'style': {
                'line-color':              'data(color)',
                'target-arrow-color':      'data(color)',
                'target-arrow-shape':      'none',
                'arrow-scale':             2,
                'width':                  f'mapData(frequency, 1, {max_edge_freq}, 2, 6)',
                'curve-style':             'bezier',
                'opacity':                f'mapData(frequency, 1, {max_edge_freq}, 0.4, 1.0)',
                'label':                  'data(label)',
                'font-size':              '9px',
                'text-background-color':   'white',
                'text-background-opacity': 0.7,
                'text-background-padding': '2px',
            }
        },
        {
            'selector': 'edge.positive',
            'style': {
                'target-arrow-shape': 'triangle',
                'target-arrow-color': 'data(color)',
                'line-style':         'solid',
                'target-arrow-fill':  'filled',
            }
        },
        {
            'selector': 'edge.negative',
            'style': {
                'target-arrow-shape':  'tee',
                'target-arrow-color':  'data(color)',
                'line-style':          'dashed',
                'line-dash-pattern':   [6, 3],
                'target-arrow-fill':   'filled',
            }
        },
    ])

    cytoscapeobj.set_layout(name='preset', animate=True)

    return cytoscapeobj

# ==================== ENUMERATION AND ANALYSIS ====================

## 1 BN per IG
def enumerate_bns_with_igs(bo, nodes: List[str], limit: int = 10) -> Tuple[pd.DataFrame, List, List]:
    """
    Enumerate subnetworks AND influence graphs from BoNesis object.
    
    Args:
        bo: BoNesis object
        nodes: List of node names
        limit: Maximum number of solutions
        
    Returns:
        Tuple of:
        - DataFrame of enumerated networks (empty if no solutions)
        - List of influence graphs
        - List of Boolean networks
    """
    from tqdm import tqdm

    try:
        gen_igs = bo.influence_graphs(
            solutions="subset-minimal",
            extra="boolean-network", 
            limit=limit
        )
        
        igs, bns = [], []
        for ig, bn in tqdm(gen_igs, desc="Enumerating IG+BN pairs"):
            igs.append(ig)
            bns.append(bn)
        len(bns)

        funcs_list = []
        for bn in bns:
            func_dict = {}
            for node in nodes:
                if node in bn:
                    func_dict[node] = str(bn[node])
            funcs_list.append(func_dict)
        
        funcs_df = pd.DataFrame(funcs_list)
        funcs_df.sort_values(by=list(funcs_df.columns), inplace=True, ignore_index=True)
        
        if not bns:
            warnings.warn("No solutions found - returning empty structures")
            return pd.DataFrame(), []
        
        return funcs_df, igs 
    
    except Exception as e:
        warnings.warn(f"Error enumerating subnetworks with IGs: {e}")
        return pd.DataFrame(), []
    

def enumerate_100bns_with_igs(bo, nodes: List[str], limit_igs: int = 10, limit_bns: int = 100) -> Tuple[pd.DataFrame, List, List]:
    """
    Enumerate influence graphs with their initial Boolean networks,
    then generate additional diverse Boolean networks independently.
    
    Args:
        bo: BoNesis object
        nodes: List of node names
        limit_igs: Maximum number of influence graphs to enumerate
        limit_bns: Maximum number of additional diverse Boolean networks to generate
        
    Returns:
        Tuple of:
        - DataFrame of enumerated networks (IG-specific + diverse)
        - List of influence graphs
        - List of all Boolean networks
    """
    from tqdm import tqdm

    try:
        print(f"Enumerating up to {limit_igs} influence graphs with their Boolean networks...")
        gen_igs = bo.influence_graphs(
            solutions="subset-minimal",
            extra="boolean-network", 
            limit=limit_igs
        )
        
        igs, bns_from_igs = [], []
        for ig, bn in tqdm(gen_igs, desc="Enumerating IG+BN pairs"):
            igs.append(ig)
            bns_from_igs.append(bn)
        
        print(f"Found {len(igs)} influence graphs with their Boolean networks")
        
        print(f"Generating up to {limit_bns} additional diverse Boolean networks...")
        bns_diverse = list(bo.diverse_boolean_networks(limit=limit_bns))
        print(f"Generated {len(bns_diverse)} additional diverse Boolean networks")
        
        all_bns = bns_from_igs + bns_diverse
        print(f"Total: {len(all_bns)} Boolean networks")

        funcs_list = []
        for bn in all_bns:
            func_dict = {}
            for node in nodes:
                if node in bn:
                    func_dict[node] = str(bn[node])
                else:
                    func_dict[node] = None
            funcs_list.append(func_dict)
        
        funcs_df = pd.DataFrame(funcs_list)
        
        if not all_bns:
            warnings.warn("No solutions found - returning empty structures")
            return pd.DataFrame(), []
        
        return funcs_df, igs
    
    except Exception as e:
        warnings.warn(f"Error enumerating subnetworks with IGs: {e}")
        return pd.DataFrame(), []


# ==================== MAIN WORKFLOW ====================

def run_full_workflow(
    compr_obs_df: pd.DataFrame,
    cells: Dict,
    influences: List,
    limit_bns: int = 10,
    limit_igs: int = 10,
    details: bool = False
) -> Dict:
    """
    Run the complete learning workflow.
    """

    def log(*args, **kwargs):
        if details:
            print(*args, **kwargs)

    results = {}

    log("\n" + "=" * 60)
    log("LEARNING RECEPTORS AND LIGANDS TOGETHER")
    log("=" * 60)

    log("\n1.A Learning the BNs with all observations...")
    bo, nodes = all_cells_synthesis(compr_obs_df, cells, influences)
    bns, igs = enumerate_100bns_with_igs(bo, nodes, limit_bns=limit_bns, limit_igs=limit_igs)
    results['n_bns'] = len(bns)
    results['n_igs'] = len(igs)
    log(f"   Found {len(bns)} functions, {len(igs)} influence graphs")

    results.update({
        'bns': bns,
        'igs': igs,
    })

    return results


if __name__ == "__main__":
    # Example usage: 2 cells, 2 receptors each, 4 ligands each (equivalent to original)
    print("Creating observation data...")

    R = {
        "A": {
            'T1': {'R1_A': 1, 'R2_A': 0},
            'T2': {'R1_A': 1, 'R2_A': 0},
            'T3': {'R1_A': 1, 'R2_A': 1},
            'T4': {'R1_A': 1, 'R2_A': 1},
            'T5': {'R1_A': 0, 'R2_A': 1},
            'T6': {'R1_A': 0, 'R2_A': 1},
            'T7': {'R1_A': 0, 'R2_A': 1},
            'T8': {'R1_A': 0, 'R2_A': 1},
        },
        "B": {
            'T1': {'R1_B': 1, 'R2_B': 0},
            'T2': {'R1_B': 1, 'R2_B': 0},
            'T3': {'R1_B': 0, 'R2_B': 1},
            'T4': {'R1_B': 0, 'R2_B': 1},
            'T5': {'R1_B': 0, 'R2_B': 0},
            'T6': {'R1_B': 0, 'R2_B': 0},
            'T7': {'R1_B': 1, 'R2_B': 1},
            'T8': {'R1_B': 1, 'R2_B': 1},
        },
    }

    L = {
        "A": {
            'T1': {'L1_A': 1, 'L2_A': 0, 'L3_A': 0, 'L4_A': 0},
            'T2': {'L1_A': 1, 'L2_A': 1, 'L3_A': 0, 'L4_A': 0},
            'T3': {'L1_A': 1, 'L2_A': 1, 'L3_A': 0, 'L4_A': 0},
            'T4': {'L1_A': 1, 'L2_A': 1, 'L3_A': 1, 'L4_A': 0},
            'T5': {'L1_A': 1, 'L2_A': 1, 'L3_A': 1, 'L4_A': 0},
            'T6': {'L1_A': 0, 'L2_A': 0, 'L3_A': 0, 'L4_A': 1},
            'T7': {'L1_A': 0, 'L2_A': 0, 'L3_A': 0, 'L4_A': 1},
            'T8': {'L1_A': 0, 'L2_A': 0, 'L3_A': 0, 'L4_A': 1},
        },
        "B": {
            'T1': {'L1_B': 1, 'L2_B': 0, 'L3_B': 0, 'L4_B': 0},
            'T2': {'L1_B': 1, 'L2_B': 1, 'L3_B': 0, 'L4_B': 0},
            'T3': {'L1_B': 1, 'L2_B': 1, 'L3_B': 0, 'L4_B': 0},
            'T4': {'L1_B': 0, 'L2_B': 0, 'L3_B': 1, 'L4_B': 0},
            'T5': {'L1_B': 0, 'L2_B': 0, 'L3_B': 1, 'L4_B': 0},
            'T6': {'L1_B': 0, 'L2_B': 0, 'L3_B': 0, 'L4_B': 0},
            'T7': {'L1_B': 0, 'L2_B': 0, 'L3_B': 0, 'L4_B': 0},
            'T8': {'L1_B': 0, 'L2_B': 0, 'L3_B': 0, 'L4_B': 1},
        },
    }

    obs_df, compr_obs_df, cells, influences, all_receptors, all_ligands = create_observation_data(R, L)

    print("\nRunning full workflow...")
    results = run_full_workflow(compr_obs_df, cells, influences, limit_bns=10, limit_igs=10)