"""
Hetnet Connectivity Search: DWPC p-values for every drug-condition pair.

Input   graph.json (full graph) or pruned_graphs/graph_stage_{PCT}pct.json
        (HECORES_PCT set, ablation RQ2), metagraph.json, nodes.tsv  -- from hpy.py
Output  {HETMAT_DIR}/path-counts/dwpc_results/all_dwpc_results_with_p.tsv
            one row per (drug, condition, metapath) with a path:
            source_id target_id source_name target_name source_degree target_degree
            path_count dwpc dwpc_raw metapath p_value adjusted_p_value
        {HETMAT_DIR}/path-counts/dwpc_results/sig_dwpc_results_path_scores.tsv
            one row per path of every pair with adjusted p <= 0.05
        HETMAT_DIR = graph.hetmat (full graph) or graph.hetmat{PCT}

Method (unchanged from the version used for the paper)
  * 200 degree-preserving permutations (hetmatpy, seed 0)
  * metapaths Drug -> Condition of length <= 4, minus a fixed exclusion list
    (Table S-metapaths); p-values for the 10 metapaths below
  * DWPC with damping 0.5, transformed to arcsinh(DWPC / mean non-zero DWPC)
    of the same graph
  * null distribution per (metapath, source degree, target degree): the
    non-zero permuted DWPCs of all pairs with those degrees, plus, for each
    real pair with a path, the permutations in which that pair has none
  * gamma-hurdle model fitted by moments; p = 0 when no permutation has a
    non-zero value for that degree group
  * Bonferroni within metapaths of the same length (x1, x6, x3)

What changed from the published script (see CHANGES_REVISION.md)
  * It ran out of the box only after fixing `collections` (never imported).
  * The published script wrote every pair of every permuted graph to disk
    (~130 MB x 200) and re-read a whole Parquet file for each of ~1.6 million
    (permutation, degree group) tasks. Here each permuted graph is reduced in
    memory to per-group counts, sums, sums of squares, minimum and maximum,
    which give the same gamma-hurdle parameters. About 45 min on 2 cores.
    Check (9 Nov 2026, full graph): of the 28,054 pairs HCS added in the
    released hcs_1.csv, 27,512 are reproduced and none is added; the other
    542 have adjusted p between 0.050 and 0.069 here, 533 of them for
    Dt>PRh>M<hPRa>CO (see CHANGES_REVISION.md).
  * percent_dwpc in the path scores divided a raw path weight by the
    transformed DWPC, so the percentages of a pair did not sum to 100; it now
    divides by the raw DWPC. A p-value of 0 no longer stops the script
    (-log10(0) is replaced by -log10 of the smallest positive double).
"""
import csv
import math
import os
import re
import shutil
import sys
import time
from multiprocessing import get_all_start_methods, get_context

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from scipy.special import gammaincc
from tqdm import tqdm

from hetnetpy import readwrite, pathtools
from hetmatpy import hetmat, degree_weight
from hetmatpy.matrix import metaedge_to_adjacency_matrix

from utils.config import PCT, HETMAT_DIR, N_PERMS, JOBS

graph_file = f"pruned_graphs/graph_stage_{PCT}pct.json" if PCT else "graph.json"
graph = readwrite.read_graph(graph_file)
metagraph = readwrite.read_metagraph("metagraph.json")
print(f"{graph_file}: {graph.n_nodes} nodes, {graph.n_edges} edges")

# Metapaths
mps_to_exclude = ['Di>CO<iDi>CO', 'Dh>CA<hDt>PRa>CO', 'Dh>CH<hDt>PRa>CO', 'Di>CO<iDt>PRa>CO', 'Di>CO<aPR<tDi>CO',
                  'Dh>PH<hDt>PRa>CO', 'Dt>PRa>CO<iDi>CO', 'Dt>PRa>CO<aPRa>CO', 'Dt>PR<tDt>PRa>CO', 'Dh>T<hDt>PRa>CO']
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
all_mps = [m.abbrev for m in metagraph.extract_metapaths(source='Drug', target='Condition', max_length=4)]
kept = [m for m in all_mps if m not in mps_to_exclude]
print(f"{len(all_mps)} metapaths, {len(kept)} kept; p-values for {len(n_metapaths)}: {list(n_metapaths)}")
missing = [m for m in n_metapaths if m not in kept]
assert not missing, f"metapaths not in this graph: {missing}"
MPS = list(n_metapaths)

out_dir = os.path.join(HETMAT_DIR, "path-counts", "dwpc_results")
os.makedirs(out_dir, exist_ok=True)


# ------------------------------------------------------------ helpers
def degrees(g, mp):
    mpo = g.metagraph.get_metapath(mp)
    _, _, s = metaedge_to_adjacency_matrix(g, mpo[0], dense_threshold=0.7)
    _, _, t = metaedge_to_adjacency_matrix(g, mpo[-1], dense_threshold=0.7)
    return np.asarray(s.sum(axis=1)).ravel().astype(int), np.asarray(t.sum(axis=0)).ravel().astype(int)


def dwpc_matrix(g, mp, damping=0.5):
    rows, cols, m = degree_weight.dwpc(g, mp, damping=damping)
    m = m.toarray() if issparse(m) else np.asarray(m, dtype=float)
    return list(rows), list(cols), m


def transform(m):
    nz = m[m != 0]
    return np.arcsinh(m / nz.mean()) if nz.size else m


# ---------------------------------------------- 1. hetmat + permutations
t0 = time.time()
perm_dir = os.path.join(HETMAT_DIR, "permutations")
if not os.path.exists(os.path.join(perm_dir, f"{N_PERMS:03d}.hetmat")):
    shutil.rmtree(HETMAT_DIR, ignore_errors=True)
    hetmat_graph = hetmat.hetmat_from_graph(graph=graph, path=HETMAT_DIR)
    hetmat_graph.permute_graph(num_new_permutations=N_PERMS)     # seed = 0 (hetmatpy default)
    os.makedirs(out_dir, exist_ok=True)
    print(f"{N_PERMS} permutations built in {time.time() - t0:.0f} s")
hetmat_graph = hetmat.HetMat(HETMAT_DIR)
perm_keys = sorted(hetmat_graph.permutations)[:N_PERMS]
assert len(perm_keys) == N_PERMS, f"found {len(perm_keys)} permutations in {perm_dir}"

# ------------------------------------------------------ 2. real graph
names = {}
for kind in ("Drug", "Condition"):
    df = pd.read_csv(hetmat_graph.get_nodes_path(kind, file_format="tsv"), sep="\t")
    names[kind] = dict(zip(df["identifier"].astype(str), df["name"]))

orig = {}
for mp in MPS:
    rows, cols, raw_m = dwpc_matrix(hetmat_graph, mp)
    z = transform(raw_m)
    _, _, pc = dwpc_matrix(hetmat_graph, mp, damping=0.0)
    sdeg, tdeg = degrees(hetmat_graph, mp)
    nz = z != 0
    r_nz, c_nz = np.nonzero(nz)
    pairs = sorted(set(zip(sdeg[r_nz], tdeg[c_nz])))
    group = np.full(z.shape, -1, dtype=np.int64)          # degree group of each pair
    for k, (ds, dt) in enumerate(pairs):
        group[np.ix_(sdeg == ds, tdeg == dt)] = k
    orig[mp] = dict(rows=rows, cols=cols, z=z, raw=raw_m, pc=pc, sdeg=sdeg, tdeg=tdeg,
                    nz=nz, n_groups=len(pairs), group=group)
    print(f"  {mp}: {nz.sum()} pairs with a path, {len(pairs)} degree groups")


# ---------------------------------- 3. null distributions (permutations)
def reduce_permutation(key):
    g = hetmat.HetMat(os.path.join(perm_dir, f"{key}.hetmat"))
    out = {}
    for mp in MPS:
        o = orig[mp]
        rows, cols, m = dwpc_matrix(g, mp)
        assert rows == o["rows"] and cols == o["cols"], "node order differs between graphs"
        z = transform(m)
        sdeg, tdeg = degrees(g, mp)
        assert (sdeg == o["sdeg"]).all() and (tdeg == o["tdeg"]).all(), "degrees not preserved"
        K, G = o["n_groups"], o["group"]
        sel = (z != 0) & (G >= 0)
        gk, v = G[sel], z[sel]
        cnt = np.bincount(gk, minlength=K)
        s = np.bincount(gk, weights=v, minlength=K)
        ss = np.bincount(gk, weights=v * v, minlength=K)
        mn = np.full(K, np.inf); np.minimum.at(mn, gk, v)
        mx = np.full(K, -np.inf); np.maximum.at(mx, gk, v)
        zeros = np.bincount(G[o["nz"] & (z == 0)], minlength=K)   # real pair with a path, none here
        out[mp] = (cnt, s, ss, mn, mx, zeros)
    return out


acc = {mp: dict(cnt=np.zeros(orig[mp]["n_groups"]), s=np.zeros(orig[mp]["n_groups"]),
                ss=np.zeros(orig[mp]["n_groups"]), zeros=np.zeros(orig[mp]["n_groups"]),
                mn=np.full(orig[mp]["n_groups"], np.inf), mx=np.full(orig[mp]["n_groups"], -np.inf))
       for mp in MPS}


def add(out):
    for mp, (cnt, s, ss, mn, mx, zeros) in out.items():
        A = acc[mp]
        A["cnt"] += cnt; A["s"] += s; A["ss"] += ss; A["zeros"] += zeros
        A["mn"] = np.minimum(A["mn"], mn); A["mx"] = np.maximum(A["mx"], mx)


t1 = time.time()
if JOBS > 1 and "fork" in get_all_start_methods():
    with get_context("fork").Pool(JOBS) as pool:
        for out in tqdm(pool.imap_unordered(reduce_permutation, perm_keys), total=len(perm_keys),
                        desc="Null distributions", file=sys.stdout):
            add(out)
else:
    for key in tqdm(perm_keys, desc="Null distributions", file=sys.stdout):
        add(reduce_permutation(key))
print(f"null distributions done in {(time.time() - t1) / 60:.1f} min")

# ------------------------------------------- 4. gamma-hurdle p-values
out_path = os.path.join(out_dir, "all_dwpc_results_with_p.tsv")
fields = ["source_id", "target_id", "source_name", "target_name", "source_degree", "target_degree",
          "path_count", "dwpc", "dwpc_raw", "metapath", "p_value", "adjusted_p_value"]
n_rows = n_sig = n_no_null = 0
with open(out_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(fields)
    for mp in MPS:
        o, A = orig[mp], acc[mp]
        cnt, n = A["cnt"], A["cnt"] + A["zeros"]
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = A["s"] / cnt
            var = (A["ss"] - cnt * mean ** 2) / (cnt - 1)       # sample variance
        var = np.where(A["mn"] == A["mx"], 0.0, var)
        beta = np.where(var > 0, mean / np.where(var > 0, var, 1), 0.0)
        alpha = mean * beta
        single = np.where((A["mn"] == A["mx"]) & (A["zeros"] == 0), A["mn"], -1.0)
        r_idx, c_idx = np.nonzero(o["nz"])
        for i, j in zip(r_idx, c_idx):
            k, d = o["group"][i, j], o["z"][i, j]
            if cnt[k] == 0:
                p = 0.0                      # no non-zero null value: empirical p = 0
                n_no_null += 1
            elif var[k] == 0:
                p = 0.0 if d > single[k] else cnt[k] / n[k]
            else:
                p = (cnt[k] / n[k]) * gammaincc(alpha[k], beta[k] * d)
            adj = min(p * n_metapaths[mp], 1.0)
            src, tgt = o["rows"][i], o["cols"][j]
            w.writerow([src, tgt, names["Drug"].get(str(src), ""), names["Condition"].get(str(tgt), ""),
                        int(o["sdeg"][i]), int(o["tdeg"][j]), o["pc"][i, j], d, o["raw"][i, j],
                        mp, p, adj])
            n_rows += 1
            n_sig += adj <= 0.05
print(f"wrote {out_path}: {n_rows} rows, {n_sig} with adjusted p <= 0.05; "
      f"{n_no_null} rows with p = 0 because no permutation had a non-zero value")


# ------------------------------------------------ 5. path scores (significant pairs)
def load_id_to_name_map(filepath):
    id_to_name = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2 and '::' in parts[0]:
                id_to_name[parts[0].split('::')[1]] = parts[1]
    return id_to_name


id_map = load_id_to_name_map('nodes.tsv')
sig_path = os.path.join(out_dir, "sig_dwpc_results_path_scores.tsv")
res = pd.read_csv(out_path, sep="\t", dtype={"source_id": str, "target_id": str})
sig = res[res.adjusted_p_value <= 0.05]
with open(sig_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=['source_id', 'target_id', 'metapath', 'path_index', 'path',
                                           'path_extended', 'percent_dwpc', 'path_score'], delimiter='\t')
    writer.writeheader()
    for row in tqdm(sig.itertuples(index=False), total=len(sig), desc="Path scores", file=sys.stdout):
        mp = metagraph.metapath_from_abbrev(row.metapath)
        paths = pathtools.paths_between(graph, ('Drug', row.source_id), ('Condition', row.target_id), mp)
        weights = [1.0 / pathtools.path_degree_product(p, damping_exponent=0.5) for p in paths]
        total = sum(weights)                             # = raw DWPC of the pair
        neg_log_p = -math.log10(max(row.adjusted_p_value, sys.float_info.min))
        for i, (path, weight) in enumerate(zip(paths, weights)):
            percent_dwpc = 100.0 * weight / total if total else 0.0
            text = path.get_unicode_str()
            extended = text
            for term in tuple(re.split(r'[←→]', text)[::2]):
                if term in id_map:
                    extended = extended.replace(term, id_map[term])
            writer.writerow({'source_id': row.source_id, 'target_id': row.target_id,
                             'metapath': row.metapath, 'path_index': i + 1, 'path': text,
                             'path_extended': extended, 'percent_dwpc': percent_dwpc,
                             'path_score': neg_log_p * percent_dwpc})
print(f"wrote {sig_path}; total time {(time.time() - t0) / 60:.1f} min")
