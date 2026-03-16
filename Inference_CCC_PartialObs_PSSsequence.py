"""
last update on the 5/02/26
Cell-Cell Communication Learning Module
========================================
This module provides functions to learn Boolean network models for cell-cell communication
through receptors and ligands using the BoNesis framework.
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
    L: Optional[Dict[str, Dict]] = None
):
    """
    Create the complete observation data for receptors and ligands across any number of cells.

    Args:
        R: Dict mapping cell name -> {timepoint -> {node_name -> value}}
           e.g. {"A": {"T1": {"R1_A": 1, "R2_A": 0}, ...}, "B": {...}, ...}
        L: Dict mapping cell name -> {timepoint -> {node_name -> value}}
           e.g. {"A": {"T1": {"L1_A": 1, ...}, ...}, "B": {...}, ...}

    Returns:
        tuple: (obs_df, compr_obs_df, cells, influences, all_receptors, all_ligands)
    """

    cell_names = list(R.keys())
    timepoints = list(next(iter(R.values())).keys())

    # Merge all observations across all cells and timepoints
    obs = {}
    for T in timepoints:
        merged = {}
        for c in cell_names:
            merged.update(R[c][T])
            merged.update(L[c][T])
        obs[T] = merged

    obs_df = pd.DataFrame.from_dict(obs, orient="index")

    all_receptors_list = [n for n in obs_df.columns if n[0] == "R"]
    all_ligands_list = [n for n in obs_df.columns if n[0] == "L"]

    # Compressed observations: receptors at odd indices (T1, T3, ...), ligands at even indices (T2, T4, ...)
    R_up = obs_df[all_receptors_list].iloc[0::2]
    R_up.index = obs_df.index[:R_up.shape[0]]
    L_up = obs_df[all_ligands_list].iloc[1::2]
    L_up.index = obs_df.index[1:L_up.shape[0] + 1]
    compr_obs_df = pd.concat([R_up, L_up], axis=1).astype("Int8")

    # Build cells structure from input dicts
    cells = {
        c: {
            "R": list(next(iter(R[c].values())).keys()),
            "L": list(next(iter(L[c].values())).keys()),
        }
        for c in cell_names
    }

    all_ligands = [node for c in cell_names for node in cells[c]["L"]]
    all_receptors = [node for c in cell_names for node in cells[c]["R"]]

    # Define influences
    influences = []
    # Each receptor can be influenced by any ligand (from any cell)
    influences += [
        (lig, rec, {"sign": 0})
        for c in cell_names for rec in cells[c]["R"]
        for d in cell_names for lig in cells[d]["L"]
    ]
    # Within each cell, link each receptor to each ligand
    influences += [
        (rec, lig, {"sign": 0})
        for c in cell_names for rec in cells[c]["R"]
        for lig in cells[c]["L"]
    ]

    return obs_df, compr_obs_df, cells, influences, all_receptors, all_ligands




# ==================== SYNTHESIS FUNCTIONS ====================


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
    receptors_A = list(cells["A"]["R"])
    ligands_A = list(cells["A"]["L"])
    receptors_B = list(cells["B"]["R"])
    ligands_B = list(cells["B"]["L"])
    
    R = receptors_A + receptors_B
    L = ligands_A + ligands_B
    nodes = R + L
    
    subnet = [(n, m, s) for (n, m, s) in influences if m in nodes]
    
    dom = bonesis.InfluenceGraph(
        subnet, 
        maxclause=8, 
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

    x1 >= x2 >= x3 >= x4 >= x5
    
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
    with bo.mutant(x4[R]):
         bo.fixed(x5[L])
    return bo, nodes


## for INCOMPLETE OBSERVATIONS 
def all_cells_synthesis_missing_timepoint(
    compr_obs_df: pd.DataFrame, 
    cells: Dict, 
    influences: List,
    missing_tp: int
):
    """
    Synthesize Boolean functions for receptors AND ligands of BOTH cells simultaneously,
    with one missing timepoint.
    
    Args:
        compr_obs_df: Compressed observation dataframe
        cells: Cell structure dictionary
        influences: List of influence tuples
        missing_tp: Which timepoint to exclude (1-5)
        
    Returns:
        tuple: (bo, nodes) - BoNesis object and list of nodes
    """
    receptors_A = list(cells["A"]["R"])
    ligands_A = list(cells["A"]["L"])
    receptors_B = list(cells["B"]["R"])
    ligands_B = list(cells["B"]["L"])
    
    R = receptors_A + receptors_B
    L = ligands_A + ligands_B
    nodes = R + L
    
    subnet = [(n, m, s) for (n, m, s) in influences if m in nodes]
    
    dom = bonesis.InfluenceGraph(
        subnet, 
        maxclause=8, 
        exact=False,
        allow_skipping_nodes=False,
        canonic=True
    )
    
    compr_obs = compr_obs_df.to_dict(orient="index")
    bo = bonesis.BoNesis(dom, compr_obs)
    
    # Create observation variables for all timepoints except the missing one
    timepoints = {}
    for i in range(1, 6):
        if i != missing_tp:
            timepoints[i] = ~bo.obs(f"T{i}")
    
    # Set up temporal ordering constraints
    sorted_tps = sorted(timepoints.keys())
    for i in range(len(sorted_tps) - 1):
        timepoints[sorted_tps[i]] >= timepoints[sorted_tps[i+1]]
    
    # LEARN RECEPTORS
    # For each available timepoint (except first and last), constrain receptors
    for tp in sorted_tps[:-1]:  # Skip the last timepoint
        if tp != missing_tp:
            with bo.mutant(timepoints[tp][L]):
                bo.fixed(timepoints[tp])
    
    # LEARN LIGANDS
    # For each pair of consecutive timepoints, learn ligand transitions
    for i in range(len(sorted_tps) - 1):
        current_tp = sorted_tps[i]
        next_tp = sorted_tps[i + 1]
        
        with bo.mutant(timepoints[current_tp][R]):
            bo.fixed(timepoints[next_tp][L])
    
    return bo, nodes






# ====================== PLOT INFLUENCE GRAPHS ====================


# with setting the position of the nodes
def plot_influence_graphs(igs, title="Influence Graph", layout='cose', node_positions=None):
    """
    Plot multiple influence graphs stacked together with frequency coloring.
    
    Parameters:
    -----------
    igs : list of networkx.DiGraph
        List of influence graphs to visualize
    title : str, optional
        Title for the visualization (displayed in stats)
    layout : str, optional
        Cytoscape layout algorithm ('cose', 'circle', 'grid', 'breadthfirst', etc.)
        Ignored if node_positions is provided.
    node_positions : dict, optional
        Dictionary mapping node names to (x, y) pixel coordinates, e.g.:
            {'A': (100, 200), 'B': (300, 150), 'C': (200, 400)}
        When provided, nodes are fixed at these positions (preset layout).
        Nodes not listed will be placed at (0, 0).

    These are good node positions for the toy model: 
    
    node_positions = {
    # Cell A — receptors (top row)
    'R1_A': (100, 50),
    'R2_A': (250, 50),
    # Cell B — receptors (top row)
    'R1_B': (425, 50),
    'R2_B': (550, 50),
    # Cell A — ligands (bottom row)
    'L1_A': (25,  300),
    'L2_A': (100, 300),
    'L3_A': (175, 300),
    'L4_A': (250, 300),
    # Cell B — ligands (bottom row)
    'L1_B': (375, 300),
    'L2_B': (450, 300),
    'L3_B': (525, 300),
    'L4_B': (600, 300),
}
    
    Returns:
    --------
    cytoscapeobj : ipycytoscape.CytoscapeWidget
        Interactive cytoscape widget
    """
    import ipycytoscape # type: ignore
    from collections import Counter
    
    # Analyze all influence graphs for edge frequency
    all_edges = []
    
    for ig in igs:
        for source, target, data in ig.edges(data=True):
            edge_tuple = (source, target, data['sign'])
            all_edges.append(edge_tuple)
    
    # Count edge frequencies
    edge_counts = Counter(all_edges)
    max_edge_freq = max(edge_counts.values()) if edge_counts else 1
    
    # Create widget with directed graph
    cytoscapeobj = ipycytoscape.CytoscapeWidget()
    cytoscapeobj.graph.directed = True
    
    # Collect all unique nodes and edges from all igs
    all_nodes = set()
    edges_with_freq = {}
    
    for ig in igs:
        all_nodes.update(ig.nodes())
        for source, target, data in ig.edges(data=True):
            edge_key = (source, target, data['sign'])
            edges_with_freq[edge_key] = edge_counts[edge_key]
    
    # Add nodes — uniform light grey, uniform size
    # Include position in node data if node_positions is provided
    for node in all_nodes:
        node_data = {'id': node, 'label': node}
        if node_positions is not None:
            x, y = node_positions.get(node, (0, 0))
            node_data['x'] = x
            node_data['y'] = y
        cytoscapeobj.graph.add_node(ipycytoscape.Node(
    data={'id': node, 'label': node},
    position={'x': x, 'y': y},
    locked=True  # prevents dragging
))
    
    # Add edges with frequency-based color
    for (source, target, sign), freq in edges_with_freq.items():
        normalized = freq / max_edge_freq
        edge_class = 'positive' if sign == 1 else 'negative'

        if sign == 1:  # Positive regulation — powder blue
            r = int(0xA0 + (0x6A - 0xA0) * normalized)
            g = int(0xC0 + (0x9F - 0xC0) * normalized)
            b = int(0xD8 + (0xC0 - 0xD8) * normalized)
            edge_color = f'#{r:02x}{g:02x}{b:02x}'
        else:  # Negative regulation — peach amber
            r = int(0xF0 + (0xE8 - 0xF0) * normalized)
            g = int(0xC0 + (0xA5 - 0xC0) * normalized)
            b = int(0x90 + (0x6A - 0x90) * normalized)
            edge_color = f'#{r:02x}{g:02x}{b:02x}'
        
        cytoscapeobj.graph.add_edge(ipycytoscape.Edge(
            data={
                'source': source, 
                'target': target, 
                'color': edge_color,
                'frequency': freq,
                'label': str(freq)
            },
            classes=edge_class
        ))
    
    cytoscapeobj.set_style([
        {
            'selector': 'node',
            'style': {
                'background-color': '#D3D3D3',
                'label': 'data(label)',
                'text-valign': 'center',
                'color': 'black',
                'text-outline-width': 2,
                'text-outline-color': 'white',
                'font-size': '14px',
                'font-weight': 'bold',
                'width': 50,
                'height': 50
            }
        },
        {
            'selector': 'edge',
            'style': {
                'line-color': 'data(color)', # '#0e0d0d',  #'data(color)',
                'target-arrow-color': 'data(color)', # '#0e0d0d',  #'data(color)',
                'target-arrow-shape': 'none',
                'arrow-scale': 2,
                'width': f'mapData(frequency, 1, {max_edge_freq}, 2, 5)',
                'curve-style': 'bezier',
                'opacity': f'mapData(frequency, 1, {max_edge_freq}, 0.4, 1.0)',
                'label': 'data(label)',
                'font-size': '10px',
                'text-background-color': 'white',
                'text-background-opacity': 0.7,
                'text-background-padding': '2px'
            }
        },
        {
            'selector': 'edge.positive',
            'style': {
                'target-arrow-shape': 'triangle',
                'target-arrow-color': 'data(color)',
                'line-style': 'solid',
                'target-arrow-fill': 'filled'
            }
        },
        {
            'selector': 'edge.negative',
            'style': {
                'target-arrow-shape': 'tee',
                'target-arrow-color': 'data(color)',
                'line-style': 'dashed',
                'line-dash-pattern': [6, 3],
                'target-arrow-fill': 'filled'
            }
        }
    ])
    
    # Use preset layout if positions are provided, otherwise use the given layout
    if node_positions is not None:
        cytoscapeobj.set_layout(name='preset', animate=True)
    else:
        cytoscapeobj.set_layout(name=layout, animate=True)
    
    return cytoscapeobj






###############
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



def plot_influence_graphs_v04(igs, title="Influence Graph",
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


def enumerate_bns_with_igs(
    bo,
    nodes: List[str],
    limit_igs: int = 10,
    limit_bns: int = 1
) -> Tuple[pd.DataFrame, List, List]:
    """
    Enumerate influence graphs with their initial Boolean networks,
    then generate additional diverse Boolean networks independently.

    Args:
        bo: BoNesis object
        nodes: List of node names
        limit_igs: Maximum number of influence graphs to enumerate
        limit_bns: Maximum number of additional diverse Boolean networks to generate.
                   If 0, no additional diverse Boolean networks are generated.

    Returns:
        Tuple of:
        - DataFrame of enumerated networks (IG-specific + diverse)
        - List of influence graphs
        - List of all Boolean networks
    """
    from tqdm import tqdm
    from typing import List, Tuple
    import pandas as pd
    import warnings

    try:
        # Step 1: Enumerate IGs with one BN each
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

        # Step 2: Generate additional diverse BNs only if requested
        if limit_bns > 0:
            print(f"Generating up to {limit_bns} additional diverse Boolean networks...")
            bns_diverse = list(bo.diverse_boolean_networks(limit=limit_bns))
            print(f"Generated {len(bns_diverse)} additional diverse Boolean networks")
        else:
            print("limit_bns=0, skipping additional diverse Boolean networks.")
            bns_diverse = []

        # Step 3: Combine all BNs
        all_bns = bns_from_igs + bns_diverse
        print(f"Total: {len(all_bns)} Boolean networks")

        # Extract Boolean functions for the requested nodes
        funcs_list = []
        for bn in all_bns:
            func_dict = {}
            for node in nodes:
                func_dict[node] = str(bn[node]) if node in bn else None
            funcs_list.append(func_dict)

        funcs_df = pd.DataFrame(funcs_list)

        if not all_bns:
            warnings.warn("No solutions found - returning empty structures")
            return pd.DataFrame(), [], []

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

    # 1. Both cells
    log("\n1.A Learning the BNs with all observations...")
    bo_all_observations, nodes_all_observations = all_cells_synthesis(compr_obs_df, cells, influences)
    bns_all_observations, igs_all_observations = enumerate_bns_with_igs(bo_all_observations, nodes_all_observations, limit_igs=limit_igs, limit_bns=limit_bns)
    results['bns_all_observations'] = bns_all_observations
    results['igs_all_observations'] = igs_all_observations
    log(f"   Found {len(bns_all_observations)} functions, {len(igs_all_observations)} influence graphs")


    # 2. Without T1
    log("\n2. Learning the BNs without T1...")
    bo_missingT1, nodes_missingT1 = all_cells_synthesis_missing_timepoint(compr_obs_df, cells, influences, missing_tp=1)
    bns_missingT1, igs_missingT1 = enumerate_bns_with_igs(bo_missingT1, nodes_missingT1, limit_bns=limit_bns, limit_igs=limit_igs)
    results['bns_missingT1'] = bns_missingT1
    results['igs_missingT1'] = igs_missingT1
    log(f"   Found {len(bns_missingT1)} functions, {len(igs_missingT1)} influence graphs")

    # 3. Without T2
    log("\n3. Learning the BNs without T2...")
    bo_missingT2, nodes_missingT2 = all_cells_synthesis_missing_timepoint(compr_obs_df, cells, influences, missing_tp=2)
    bns_missingT2, igs_missingT2 = enumerate_bns_with_igs(bo_missingT2, nodes_missingT2, limit_bns=limit_bns, limit_igs=limit_igs)
    results['bns_missingT2'] = bns_missingT2
    results['igs_missingT2'] = igs_missingT2
    log(f"   Found {len(bns_missingT2)} functions, {len(igs_missingT2)} influence graphs")

    # 4. Without T3
    log("\n4. Learning the BNs without T3...")
    bo_missingT3, nodes_missingT3 = all_cells_synthesis_missing_timepoint(compr_obs_df, cells, influences, missing_tp=3)
    bns_missingT3, igs_missingT3 = enumerate_bns_with_igs(bo_missingT3, nodes_missingT3, limit_bns=limit_bns, limit_igs=limit_igs)
    results['bns_missingT3'] = bns_missingT3
    results['igs_missingT3'] = igs_missingT3
    log(f"   Found {len(bns_missingT3)} functions, {len(igs_missingT3)} influence graphs")
    
    # 5. Without T4
    log("\n5. Learning the BNs without T4...")
    bo_missingT4, nodes_missingT4 = all_cells_synthesis_missing_timepoint(compr_obs_df, cells, influences, missing_tp=4)
    bns_missingT4, igs_missingT4 = enumerate_bns_with_igs(bo_missingT4, nodes_missingT4, limit_bns=limit_bns, limit_igs=limit_igs)
    results['bns_missingT4'] = bns_missingT4
    results['igs_missingT4'] = igs_missingT4
    log(f"   Found {len(bns_missingT4)} functions, {len(igs_missingT4)} influence graphs")

    # 6. Without T5
    log("\n6. Learning the BNs without T5...")
    bo_missingT5, nodes_missingT5 = all_cells_synthesis_missing_timepoint(compr_obs_df, cells, influences, missing_tp=5)
    bns_missingT5, igs_missingT5 = enumerate_bns_with_igs(bo_missingT5, nodes_missingT5, limit_bns=limit_bns, limit_igs=limit_igs)
    results['bns_missingT5'] = bns_missingT5
    results['igs_missingT5'] = igs_missingT5
    log(f"   Found {len(bns_missingT5)} functions, {len(igs_missingT5)} influence graphs") 
 


    return results


if __name__ == "__main__":
    # Example usage
    print("Creating observation data...")
    obs_df, compr_obs_df, cells, influences, all_receptors, all_ligands = create_observation_data()
    
    print("\nRunning full workflow...")
    results = run_full_workflow(compr_obs_df, cells, influences, limit_igs=500, limit_bns=10, details=False)
    
