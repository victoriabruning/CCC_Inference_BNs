"""
Cell-cell communication Boolean networks inference for 2 cell types 
========================================
These are the functions to infer the Boolean networks of cell-cell communication
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


def ccc_bns_synthesis(compr_obs_df: pd.DataFrame, cells: Dict, influences: List):
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

    x1 >= x2 >= x3 >= x4 >= x5 
    
    # LEARN RECEPTORS
    with bo.mutant(x2[L]): 
        bo.fixed(x2)
    with bo.mutant(x3[L]):
        bo.fixed(x3)
    with bo.mutant(x4[L]):
        bo.fixed(x4)
    with bo.mutant(x5[L]):
        bo.fixed(x5)

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

    x1 >= x2 >= x3 >= x4 >= x5 
    

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




# ==================== ENUMERATION AND ANALYSIS ====================

## 1 IG => 1 BN
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

        # Extract Boolean functions for the requested nodes
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
    



def enumerate_100bns_with_igs(
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
    bo, nodes = ccc_bns_synthesis(compr_obs_df, cells, influences)
    bns, igs = enumerate_100bns_with_igs(bo, nodes, limit_igs=limit_igs, limit_bns=limit_bns)
    results['bns'] = bns
    results['igs'] = igs
    results['n_both_funcs'] = len(bns)
    results['n_igs_both'] = len(igs)
    log(f"   Found {len(bns)} functions, {len(igs)} influence graphs")

    # RECAP
    print(f"""
    Number of functions, inferred with all observations : {results.get('n_both_funcs', 'N/A')}
    Number of influence graphs, inferred with all observations: {results.get('n_igs_both', 'N/A')}
            """)

    # Store outputs
    results.update({
        'bns': bns,
        'igs': igs
    })

    return results


if __name__ == "__main__":
    # Example usage
    print("Creating observation data...")
    obs_df, compr_obs_df, cells, influences, all_receptors, all_ligands = create_observation_data()
    
    print("\nRunning full workflow...")
    results = run_full_workflow(compr_obs_df, cells, influences, limit_igs=500, limit_bns=10, details=False)
    
