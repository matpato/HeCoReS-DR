import hetnetpy
from hetnetpy import readwrite, hetnet
from utils.utils import open_json
import pandas
import matplotlib
import matplotlib.backends.backend_pdf
import seaborn

import json
import random
import os
from copy import deepcopy

# Metagraph creation
metagraph = hetnet.MetaGraph.from_edge_tuples([
    ('Drug', 'TherapeuticGroup', 'has_therapeutic_group', 'forward'),
    ('Drug', 'PharmacologicalGroup', 'has_pharmacological_group', 'forward'),
    ('Drug', 'ChemicalGroup', 'has_chemical_group', 'forward'),
    ('Drug', 'Condition', 'indicated_for', 'forward'),
    ('Drug', 'Protein', 'targets', 'forward'),
    ('Protein', 'Condition', 'associated_with', 'forward'),
    ('Protein', 'CellularComponent', 'has_component', 'forward'),
    ('Protein', 'MolecularFunction', 'has_function', 'forward'),
    ('Protein', 'BiologicalProcess', 'has_process', 'forward'),
    ('Drug', 'CategoryDrug', 'has_category', 'forward')
])

print(metagraph.n_nodes) # 10
print(metagraph.n_edges) # 10

# Graph creation (nodes)
node_data = open_json('graph_nodes.json')

graph = hetnet.Graph(metagraph)

for metanode, identifiers in node_data.items():
    for ident in identifiers:
        graph.add_node(kind=metanode, identifier=ident[0], name=ident[1])

# Graph creation (edges)
def get_node_kind(identifier):
    for kind, nodes in node_data.items():
        for node in nodes:
            if node[0] == identifier:
                return kind
            
with open('graph_triples.txt') as f:
    for line in f:
        head_id, kind, tail_id = line.strip().split()
        
        # Get source and target node objects
        source = (get_node_kind(head_id), head_id)
        target = (get_node_kind(tail_id), tail_id)
        
        if source is None or target is None:
            print(f"Skipping edge: {head_id} {kind} {tail_id} — missing node")
            continue
            
        graph.add_edge(source, target, kind.lower(), 'forward')

print(graph.n_nodes) # 20270
print(graph.n_edges) # 92157

readwrite.write_graph(graph, "graph.json")
readwrite.write_metagraph(metagraph, "metagraph.json")
readwrite.write_nodetable(graph, "nodes.tsv")
readwrite.write_sif(graph, "edges.sif")

def get_metanode_df(graph):
    rows = list()
    for metanode, nodes in graph.get_metanode_to_nodes().items():
        series = pandas.Series()
        series["metanode"] = metanode
        series["abbreviation"] = metanode.abbrev
        metaedges = set()
        for metaedge in metanode.edges:
            metaedges |= {metaedge, metaedge.inverse}
        series["metaedges"] = sum(not metaedge.inverted for metaedge in metaedges)
        series["nodes"] = len(nodes)
        series["unconnected_nodes"] = sum(
            not any(node.edges.values()) for node in nodes
        )
        rows.append(series)

    metanode_df = pandas.DataFrame(rows).sort_values("metanode")
    return metanode_df

def get_metaedge_df(graph):
    rows = list()
    for metaedge, edges in graph.get_metaedge_to_edges(exclude_inverts=True).items():
        series = pandas.Series()
        series["metaedge"] = str(metaedge)
        series["abbreviation"] = metaedge.abbrev
        series["edges"] = len(edges)
        series["source_nodes"] = len({edge.source for edge in edges})
        series["target_nodes"] = len({edge.target for edge in edges})
        rows.append(series)
    metaedge_df = pandas.DataFrame(rows).sort_values("metaedge")
    return metaedge_df

def get_degrees_for_metanode(graph, metanode):
    """
    Return a dataframe that reports the degree of each metaedge for
    each node of kind metanode.
    """
    metanode_to_nodes = graph.get_metanode_to_nodes()
    nodes = metanode_to_nodes.get(metanode, [])
    rows = list()
    for node in nodes:
        for metaedge, edges in node.edges.items():
            rows.append((node.identifier, node.name, str(metaedge), len(edges)))
    df = pandas.DataFrame(rows, columns=["node_id", "node_name", "metaedge", "degree"])
    return df.sort_values(["node_name", "metaedge"])

def plot_degrees_for_metanode(graph, metanode, col_wrap=2, facet_height=4):
    """
    Plots histograms of the degree distribution of each metaedge
    incident to the metanode. Each metaedge receives a facet in
    a seaborn.FacetGrid.
    """
    degree_df = get_degrees_for_metanode(graph, metanode)
    grid = seaborn.FacetGrid(
        degree_df,
        col="metaedge",
        sharex=False,
        sharey=False,
        col_wrap=col_wrap,
        height=facet_height,
    )
    grid.map(seaborn.histplot, "degree", kde=False)
    grid.set_titles("{col_name}")
    return grid

def plot_degrees(graph, path):
    """
    Creates a multipage pdf with a page for each metanode showing degree
    distributions.
    """
    # Temporarily disable `figure.max_open_warning`
    max_open = matplotlib.rcParams["figure.max_open_warning"]
    matplotlib.rcParams["figure.max_open_warning"] = 0
    pdf_pages = matplotlib.backends.backend_pdf.PdfPages(path)
    for metanode in graph.metagraph.get_nodes():
        grid = plot_degrees_for_metanode(graph, metanode)
        grid.savefig(pdf_pages, format="pdf")
    pdf_pages.close()
    matplotlib.rcParams["figure.max_open_warning"] = max_open

print(get_metanode_df(graph))
print(get_metaedge_df(graph))
plot_degrees(graph, "degrees.pdf")

def prune_edges_cumulative(file_path, percentages, output_dir, seed=None):
    if seed is not None:
        random.seed(seed)
    
    # Load the original graph
    with open(file_path, "r", encoding="utf-8") as f:
        original_graph = json.load(f)

    # Group original edges by kind for reference
    orig_edges_by_kind = {}
    for e in original_graph["edges"]:
        orig_edges_by_kind.setdefault(e["kind"], []).append(e)

    # Working copy of current graph
    current_graph = deepcopy(original_graph)

    for stage_idx, target_percent in enumerate(percentages, start=1):
        new_edges = []
        
        # Process each kind separately
        for kind, orig_edges in orig_edges_by_kind.items():
            # Skip indicated_for edges entirely
            if kind == "indicated_for":
                # Keep all that still exist
                current_kind_edges = [e for e in current_graph["edges"] if e["kind"] == kind]
                new_edges.extend(current_kind_edges)
                continue

            original_count = len(orig_edges)
            target_remove_count = int(original_count * target_percent / 100)

            # Current edges of this kind
            current_kind_edges = [e for e in current_graph["edges"] if e["kind"] == kind]
            removed_so_far = original_count - len(current_kind_edges)

            # How many more to remove this stage
            to_remove_now = max(0, target_remove_count - removed_so_far)

            if to_remove_now > 0 and current_kind_edges:
                remove_indices = set(random.sample(range(len(current_kind_edges)), min(to_remove_now, len(current_kind_edges))))
                for idx, e in enumerate(current_kind_edges):
                    if idx not in remove_indices:
                        new_edges.append(e)
            else:
                new_edges.extend(current_kind_edges)

        # Update current graph's edges
        current_graph["edges"] = new_edges

        # Remove orphan nodes
        referenced_ids = set()
        for edge in new_edges:
            referenced_ids.add(tuple(edge["source_id"]))
            referenced_ids.add(tuple(edge["target_id"]))

        current_graph["nodes"] = [
            node for node in current_graph["nodes"]
            if (node["kind"], node["identifier"]) in referenced_ids
        ]

        # Save stage output
        os.makedirs(output_dir, exist_ok=True)
        stage_file = os.path.join(output_dir, f"graph_stage_{target_percent}pct.json")
        with open(stage_file, "w", encoding="utf-8") as f:
            json.dump(current_graph, f, indent=2, ensure_ascii=False)

        print(f"Stage {stage_idx}: Removed {target_percent}% of original edges → saved to {stage_file}")

        # --- Reporting: count edges per kind in the current graph ---
        kind_counts = {}
        for e in current_graph["edges"]:
            kind_counts[e["kind"]] = kind_counts.get(e["kind"], 0) + 1

        print("  Remaining edges by kind:")
        for kind, count in sorted(kind_counts.items()):
            print(f"    {kind}: {count}")
        print("-" * 50)

# Remove progressively: 20%, then 40%, then 60%, then 80%
prune_edges_cumulative(
    file_path="graph.json",
    percentages=[20, 40, 60, 80],
    output_dir="pruned_graphs",
    seed=0
)