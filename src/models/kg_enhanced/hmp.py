from collections import defaultdict
import itertools
import numpy as np
import pandas as pd
import scipy.sparse
import os
import pickle
import csv
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from scipy.special import gammaincc
from statistics import mean, variance
import glob
import math
import re
from functools import lru_cache

from hetmatpy.matrix import metaedge_to_adjacency_matrix
from hetnetpy import readwrite, pathtools
from hetmatpy import hetmat, degree_weight

# pct = 20

graph = readwrite.read_graph("graph.json")
# graph = readwrite.read_graph(f"pruned_graphs/graph_stage_{pct}pct.json") # when working with the pruned graphs
metagraph = readwrite.read_metagraph("metagraph.json")

hetmat_graph = hetmat.hetmat_from_graph(graph=graph, path=f"graph.hetmat")
n_perms = 200
perm_data = hetmat_graph.permute_graph(num_new_permutations=n_perms)
perm_dict = hetmat_graph.permutations

hetmat_graph.path_counts_directory.mkdir()
for k in perm_dict.keys():
    perm_dict[k].path_counts_directory.mkdir()

# Metapaths
length = 4
metapaths = metagraph.extract_metapaths(
    source='Drug',
    target='Condition',
    max_length=length
)

mps = {}
for i, metapath in enumerate(metapaths):
    mp = metagraph.metapath_from_abbrev(str(metapath))
    mps[metapath.abbrev] = [str(e) for e in mp.edges]

mps_to_exclude = ['Di>CO<iDi>CO', 'Dh>CA<hDt>PRa>CO', 'Dh>CH<hDt>PRa>CO', 'Di>CO<iDt>PRa>CO', 'Di>CO<aPR<tDi>CO',
                  'Dh>PH<hDt>PRa>CO', 'Dt>PRa>CO<iDi>CO', 'Dt>PRa>CO<aPRa>CO', 'Dt>PR<tDt>PRa>CO', 'Dh>T<hDt>PRa>CO']
for mp in mps_to_exclude:
    del mps[mp]

print(mps.keys())
print(len(mps.keys()))

for mp in mps.keys():
    rows, cols, dwpc = degree_weight.dwpc(hetmat_graph, mp)
    path = f"graph.hetmat/path-counts/dwpc-0.5/{mp}"
    hetmat.save_matrix(dwpc, path)

    rows_0, cols_0, dwpc_0 = degree_weight.dwpc(hetmat_graph, mp, damping=0.0)
    path = f"graph.hetmat/path-counts/dwpc-0.0/{mp}"
    hetmat.save_matrix(dwpc_0, path)

    for k in perm_dict.keys():
        rows, cols, dwpc = degree_weight.dwpc(perm_dict[k], mp)
        path = f"graph.hetmat/permutations/{k}.hetmat/path-counts/dwpc-0.5/{mp}"
        hetmat.save_matrix(dwpc, path)

        rows_0, cols_0, dwpc_0 = degree_weight.dwpc(perm_dict[k], mp, damping=0.0)
        path = f"graph.hetmat/permutations/{k}.hetmat/path-counts/dwpc-0.0/{mp}"
        hetmat.save_matrix(dwpc_0, path)

def dwpc_to_degrees(graph, metapath, damping=0.5, ignore_zeros=True, ignore_redundant=True):

    metapath = graph.metagraph.get_metapath(metapath)
    _, _, source_adj_mat = metaedge_to_adjacency_matrix(
        graph, metapath[0], dense_threshold=0.7
    )
    _, _, target_adj_mat = metaedge_to_adjacency_matrix(
        graph, metapath[-1], dense_threshold=0.7
    )
    source_degrees = source_adj_mat.sum(axis=1).flat
    target_degrees = target_adj_mat.sum(axis=0).flat
    del source_adj_mat, target_adj_mat

    source_path = graph.get_nodes_path(metapath.source(), file_format="tsv")
    source_node_df = pd.read_csv(source_path, sep="\t")
    source_node_names = list(source_node_df["name"])

    target_path = graph.get_nodes_path(metapath.target(), file_format="tsv")
    target_node_df = pd.read_csv(target_path, sep="\t")
    target_node_names = list(target_node_df["name"])

    row_names, col_names, dwpc_matrix = graph.read_path_counts(
        metapath, "dwpc", damping
    )
    mean_nonzero = dwpc_matrix[dwpc_matrix != 0].mean()
    dwpc_matrix = np.arcsinh(dwpc_matrix / mean_nonzero)
    if scipy.sparse.issparse(dwpc_matrix):
        dwpc_matrix = dwpc_matrix.toarray()

    _, _, path_count = graph.read_path_counts(metapath, "dwpc", 0.0)
    if scipy.sparse.issparse(path_count):
        path_count = path_count.toarray()

    if ignore_redundant and metapath.is_symmetric():
        pairs = itertools.combinations_with_replacement(range(len(row_names)), 2)
    else:
        pairs = itertools.product(range(len(row_names)), range(len(col_names)))
    for row_ind, col_ind in pairs:
        dwpc_value = dwpc_matrix[row_ind, col_ind]
        if ignore_zeros and dwpc_value == 0:
            continue
        row = {
            "source_id": row_names[row_ind],
            "target_id": col_names[col_ind],
            "source_name": source_node_names[row_ind],
            "target_name": target_node_names[col_ind],
            "source_degree": source_degrees[row_ind],
            "target_degree": target_degrees[col_ind],
            "path_count": path_count[row_ind, col_ind],
            "dwpc": dwpc_value,
        }
        yield collections.OrderedDict(row)

def compute_dwpc_results(hetmat_graph, mps):
    """Compute DWPC results for a given hetmat graph and metapaths."""
    all_results = []
    for mp_str in tqdm(mps, desc="Computing DWPCs for metapaths"):
        results = list(dwpc_to_degrees(hetmat_graph, mp_str))
        if results:
            df = pd.DataFrame(results)
            df["metapath"] = mp_str
            all_results.append(df)

    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return None

def save_dwpc_results(df, output_path):
    """Save DWPC results to a TSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, sep="\t", index=False)

def process_original_graph(hetmat_graph, mps, output_path):
    """Process and save DWPC results for the original graph."""
    final_df = compute_dwpc_results(hetmat_graph, mps)
    if final_df is not None:
        save_dwpc_results(final_df, output_path)

def process_permuted_graphs(base_dir, n_perms, perm_dict, mps):
    """Process and save DWPC results for permuted graphs."""
    for i in tqdm(range(1, n_perms + 1), desc="Processing permuted graphs"):
        graph_key = f"{i:03d}"
        hetmat_graph = perm_dict[graph_key]
        perm_path = os.path.join(base_dir, f"{graph_key}.hetmat")
        output_path = os.path.join(perm_path, "path-counts", "dwpc_results", "all_dwpc_results.tsv")

        final_df = compute_dwpc_results(hetmat_graph, mps)
        if final_df is not None:
            save_dwpc_results(final_df, output_path)

# Original graph
process_original_graph(
    hetmat_graph=hetmat_graph,
    mps=mps,
    output_path=f"graph.hetmat/path-counts/dwpc_results/all_dwpc_results.tsv"
)

# Permuted graphs
process_permuted_graphs(
    base_dir=f"graph.hetmat/permutations",
    n_perms=n_perms,
    perm_dict=perm_dict,
    mps=mps
)

data_zeros = dict()
input_file = f'/workspace/hcs/graph.hetmat/path-counts/dwpc_results/all_dwpc_results.tsv'
with open(input_file, newline='', encoding='utf-8') as f:
    reader = list(csv.DictReader(f, delimiter='\t'))
    for row in reader:
        key = (row['source_degree'], row['target_degree'], row['source_id'], row['target_id'], row['metapath'])
        data_zeros[key] = 0

base_dir = f'/workspace/hcs/graph.hetmat/permutations'
print("Checking entries against permutations...")
for i in tqdm(range(1, n_perms + 1), desc="Processing permutations"):
    folder = f"{i:03d}.hetmat"
    file_path = os.path.join(base_dir, folder, "path-counts", "dwpc_results", "all_dwpc_results.tsv")
    df = pd.read_csv(file_path, sep="\t", dtype=str)
    key_set = set(zip(df['source_degree'], df['target_degree'], df['source_id'], df['target_id'], df['metapath']))
    for key in data_zeros:
        if key not in key_set:
            data_zeros[key] += 1

with open("data_zeros.pkl", 'wb') as f:
    pickle.dump(data_zeros, f)

mp_pair_counts = defaultdict(set)
with open(f'/workspace/hcs/graph.hetmat/path-counts/dwpc_results/all_dwpc_results.tsv', newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        metapath = row['metapath']
        pair = (int(row['source_degree']), int(row['target_degree']))
        mp_pair_counts[metapath].add(pair)

mp_pair_counts = {k: list(v) for k, v in mp_pair_counts.items()}

for i in range(1, n_perms + 1):
    folder = f"{i:03d}.hetmat"
    tsv_path = os.path.join(base_dir, folder, "path-counts", "dwpc_results", "all_dwpc_results.tsv")
    parquet_path = tsv_path.replace(".tsv", ".parquet")

    if os.path.exists(tsv_path):
        try:
            df = pd.read_csv(tsv_path, sep="\t", usecols=["source_id", "target_id", "source_degree", "target_degree", "metapath", "dwpc"])
            df.to_parquet(parquet_path)
            print(f"Converted {tsv_path} to Parquet")
        except Exception as e:
            print(f"Failed to convert {tsv_path}: {e}")

def combine_dwpcs_parquet(args):
    """Read Parquet file and return DWPCs matching the degree pair and metapath."""
    i, degree_pair, metapath = args
    folder = f"{i:03d}.hetmat"
    file_path = os.path.join(base_dir, folder, "path-counts", "dwpc_results", "all_dwpc_results.parquet")

    if not os.path.exists(file_path):
        return (degree_pair, [])

    try:
        df = pd.read_parquet(file_path)
        match = df[
            (df["source_degree"] == degree_pair[0]) &
            (df["target_degree"] == degree_pair[1]) &
            (df["metapath"] == metapath)
        ]
        dwpcs = match["dwpc"].tolist()
        return (degree_pair, dwpcs)
    except Exception:
        return (degree_pair, [])

def dwpc_null_distributions(n_perms, degree_pairs, metapath):
    results = defaultdict(list)
    tasks = [(i, dp, metapath) for dp in degree_pairs for i in range(1, n_perms + 1)]

    with ProcessPoolExecutor() as executor:
        for dp, dwpcs in tqdm(executor.map(combine_dwpcs_parquet, tasks), total=len(tasks), desc=f"Processing {metapath}"):
            if dwpcs:
                results[dp].extend(dwpcs)

    path = f'dwpc_null_counts_{metapath}.pkl'
    with open(path, 'wb') as f:
        pickle.dump(dict(results), f)

metapaths = ['Dt>PRa>CO', 'Dh>CA<hDi>CO', 'Dh>CH<hDi>CO', 'Di>CO<aPRa>CO', 'Dh>PH<hDi>CO', 'Dt>PR<tDi>CO', 'Dh>T<hDi>CO', 
             'Dt>PRh>B<hPRa>CO', 'Dt>PRh>CE<hPRa>CO', 'Dt>PRh>M<hPRa>CO']

for mp in metapaths:
    degree_pairs = mp_pair_counts[mp]
    dwpc_null_distributions(n_perms, degree_pairs, mp)

with open("data_zeros.pkl", 'rb') as f1:
    zeros = pickle.load(f1)
    grouped = defaultdict(int)

    for key, value in zeros.items():
        new_key = (key[0], key[1], key[-1])
        grouped[new_key] += value

    grouped_dict = dict(grouped)

for mp in tqdm(metapaths, desc="Processing metapaths"):
    with open(f"dwpc_null_counts_{mp}.pkl", 'rb') as f:
        data = pickle.load(f)

    for k in tqdm(data.keys(), desc=f"Updating {mp}", leave=False):
        n_zeros = grouped_dict[(str(k[0]), str(k[1]), mp)]
        data[k].extend([0] * n_zeros)

    with open(f"dwpc_null_counts_{mp}.pkl", 'wb') as f:
        pickle.dump(data, f)

def compute_gamma_params(null_counts_file):
    with open(null_counts_file, 'rb') as file:
        data = pickle.load(file)
        data_gamma = dict()

        for degree_pair, null_dwpcs in data.items():
            n = len(null_dwpcs)
            nz_dwpcs = [dwpc for dwpc in null_dwpcs if dwpc != 0]
            nnz = len(nz_dwpcs)
            mean_nz = mean(nz_dwpcs)
            var_nz = variance(nz_dwpcs)
            if var_nz == 0:
                beta = 0
            else:
                beta = mean_nz / var_nz
            alpha = mean_nz * beta
            if len(set(null_dwpcs)) == 1 and null_dwpcs[0] != 0:
                single_dwpc = null_dwpcs[0]
                data_gamma[(degree_pair)] = (n, nnz, alpha, beta, var_nz, single_dwpc)
            else:
                data_gamma[(degree_pair)] = (n, nnz, alpha, beta, var_nz, -1)

    return data_gamma

for mp in tqdm(metapaths, desc="Processing gamma parameters' computation"):
    read_file = f"dwpc_null_counts_{mp}.pkl"
    write_file = f"gamma_{mp}.pkl"

    data_gamma = compute_gamma_params(read_file)

    with open(write_file, 'wb') as f:
        pickle.dump(data_gamma, f)

def calculate_p_value(n, nnz, alpha, beta, dwpc, var_nz, single_dwpc):
    if nnz == 0:
        # No nonzero DWPCs are found in the permuted network, but paths are observed in the true network
        # When all null DWPCs are zero but the observed DWPC is positive, P empiric = 0.
        return 0.0
    if var_nz == 0:
        # When all nonzero null DWPCs have the same positive value (standard deviation = 0)
        if dwpc > single_dwpc:
            return 0.0
        else:
            return nnz / n

    hurdle = nnz / n
    return hurdle * gammaincc(alpha, beta * dwpc)

gamma_cache = {
    path.replace("gamma_", "").replace(".pkl", ""): pickle.load(open(path, 'rb'))
    for path in glob.glob("gamma_*.pkl")
}

n_metapaths = {
    'Dt>PRa>CO': 1,
    'Dh>CA<hDi>CO': 6,
    'Dh>CH<hDi>CO': 6,
    'Di>CO<aPRa>CO': 6,
    'Dh>PH<hDi>CO': 6,
    'Dh>T<hDi>CO': 6,
    'Dt>PR<tDi>CO': 6,
    'Dt>PRh>B<hPRa>CO': 3,
    'Dt>PRh>CE<hPRa>CO': 3, 
    'Dt>PRh>M<hPRa>CO': 3
}

input_path = f'/workspace/hcs/graph.hetmat/path-counts/dwpc_results/all_dwpc_results.tsv'
output_path = f'/workspace/hcs/graph.hetmat/path-counts/dwpc_results/all_dwpc_results_with_p.tsv'

# Bonferroni
with open(input_path, newline='', encoding='utf-8') as infile, \
     open(output_path, 'w', newline='', encoding='utf-8') as outfile:

    reader = csv.DictReader(infile, delimiter='\t')
    fieldnames = reader.fieldnames + ['p_value', 'adjusted_p_value']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter='\t')
    writer.writeheader()

    rows_to_write = []

    for row in tqdm(reader, desc="Processing DWPC rows"):
        s_degree = int(row['source_degree'])
        t_degree = int(row['target_degree'])
        dwpc = float(row['dwpc'])
        metapath = row['metapath']

        if metapath not in gamma_cache:
            continue

        data = gamma_cache[metapath]
        n, nnz, alpha, beta, var_nz, single_dwpc = data.get((s_degree, t_degree), (0, 0, 0, 0, 0, 0))
        p_value = calculate_p_value(n, nnz, alpha, beta, dwpc, var_nz, single_dwpc)
        row['p_value'] = p_value
        adjusted_p_value = p_value * n_metapaths[metapath]
        adjusted_p_value = min(adjusted_p_value, 1.0)
        row['adjusted_p_value'] = adjusted_p_value

        rows_to_write.append(row)
        if len(rows_to_write) >= 10000:
            writer.writerows(rows_to_write)
            rows_to_write = []

    if rows_to_write:
        writer.writerows(rows_to_write)

def load_id_to_name_map(filepath):
    id_to_name = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        next(f)  # Skip the header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            full_id, name = parts[0], parts[1]
            # Extract the part after '::'
            if '::' in full_id:
                short_id = full_id.split('::')[1]
                id_to_name[short_id] = name
    return id_to_name

def get_name_by_id(id_short, id_to_name_map):
    return id_to_name_map.get(id_short, None)

@lru_cache(maxsize=None)
def get_name_by_id_cached(term):
    return get_name_by_id(term, id_map)

id_map = load_id_to_name_map('nodes.tsv')

input_path = f'/workspace/hcs/graph.hetmat/path-counts/dwpc_results/all_dwpc_results_with_p.tsv'
output_path = f'/workspace/hcs/graph.hetmat/path-counts/dwpc_results/sig_dwpc_results_path_scores.tsv'
with open(input_path, newline='', encoding='utf-8') as infile, \
     open(output_path, 'w', newline='', encoding='utf-8') as outfile:

    reader = csv.DictReader(infile, delimiter='\t')

    fieldnames = ['source_id', 'target_id', 'metapath', 'path_index', 'path', 'path_extended', 'percent_dwpc', 'path_score']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter='\t')
    writer.writeheader()

    for row in tqdm(reader, desc="Processing significant DWPC rows"):
        adj_p = float(row['adjusted_p_value'])
        if adj_p <= 0.05:
            source_id = row['source_id']
            target_id = row['target_id']
            source = 'Drug', source_id
            target = 'Condition', target_id
            dwpc = float(row['dwpc'])
            metapath = row['metapath']

            mp = metagraph.metapath_from_abbrev(metapath)
            paths = pathtools.paths_between(graph, source, target, mp)

            degree_products = [pathtools.path_degree_product(p, damping_exponent=0.5) for p in paths]
            path_weights = [1.0 / dp for dp in degree_products]

            for i, (path, weight) in enumerate(zip(paths, path_weights)):
                percent_dwpc = (weight / dwpc) * 100
                path_score = -math.log10(adj_p) * percent_dwpc
                path = path.get_unicode_str()
                parts = re.split(r'[←→]', path)
                ids = tuple(parts[::2])
                path_extended = path
                for term in ids:
                    name = get_name_by_id(term, id_map)
                    if name:
                        path_extended = path_extended.replace(term, name)

                writer.writerow({
                    'source_id': source_id,
                    'target_id': target_id,
                    'metapath': metapath,
                    'path_index': i+1,
                    'path': path,
                    'path_extended': path_extended,
                    'percent_dwpc': percent_dwpc,
                    'path_score': path_score
                })
