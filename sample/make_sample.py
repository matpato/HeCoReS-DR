"""
Build the HeCoReS-DR sample: a real subset of the pipeline's starting files.

The sample keeps the drugs of one ATC anatomical group (default: C,
cardiovascular system), their indications, the part of the knowledge graph
around them and the clinical trials that involve them. It is cut with the same
rule as step 8 (every drug keeps at least 2 indications, every condition at
least 2 drugs), so it can be run through steps 11-16 like the full data.

Usage (from the HeCoReS-DR main folder):

    python3 sample/make_sample.py --full data/work --out sample/inputs [--atc C]

--full must hold the files the pipeline has after step 8:
    drugs_with_conditions_final.json, conditions.json (filtered, 722 conditions),
    graph_nodes.json, graph_triples.txt, clinical_trials.json

What is kept from each file is described in sample/README.md.
"""
import argparse
import json
import os
from collections import defaultdict

DRUG_EDGES = ("HAS_THERAPEUTIC_GROUP", "HAS_PHARMACOLOGICAL_GROUP", "HAS_CHEMICAL_GROUP",
              "HAS_CATEGORY", "TARGETS")
PROTEIN_EDGES = ("HAS_PROCESS", "HAS_FUNCTION", "HAS_COMPONENT")


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
        f.write("\n")


def prune(treats, min_conditions=2, min_drugs=2):
    """Same rule as prune() in src/kg/processing_for_neo4j.py (step 8)."""
    treats = {d: list(c) for d, c in treats.items()}
    while True:
        bad_drugs = [d for d, c in treats.items() if len(c) < min_conditions]
        for d in bad_drugs:
            del treats[d]
        cond_to_drugs = defaultdict(set)
        for d, cs in treats.items():
            for c in cs:
                cond_to_drugs[c].add(d)
        bad_conds = {c for c, ds in cond_to_drugs.items() if len(ds) < min_drugs}
        for d in treats:
            treats[d] = [c for c in treats[d] if c not in bad_conds]
        if not bad_drugs and not bad_conds:
            return treats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", default="data/work", help="folder with the full starting files")
    ap.add_argument("--out", default="sample/inputs", help="folder for the sample files")
    ap.add_argument("--atc", default="C", help="ATC anatomical group (first letter of the code)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    drugs = load(os.path.join(a.full, "drugs_with_conditions_final.json"))
    conditions = load(os.path.join(a.full, "conditions.json"))
    nodes = load(os.path.join(a.full, "graph_nodes.json"))
    trials = load(os.path.join(a.full, "clinical_trials.json"))
    with open(os.path.join(a.full, "graph_triples.txt"), encoding="utf-8") as f:
        triples = [line.split() for line in f if line.strip()]

    # 1. drugs of the ATC group (from the graph's therapeutic-group edges), pruned as in step 8
    in_group = {h for h, k, t in triples if k == "HAS_THERAPEUTIC_GROUP" and t.startswith(a.atc)}
    treats = prune({d: info["treats"] for d, info in drugs.items() if d in in_group})
    D = set(treats)
    C = {c for cs in treats.values() for c in cs}
    pairs = {(d, c) for d, cs in treats.items() for c in cs}

    # 2. graph around them: the drugs' own edges, their indications, the proteins
    #    associated with their conditions, and the GO terms of every protein kept
    kept = []
    for h, k, t in triples:
        if (k == "INDICATED_FOR" and (h, t) in pairs) or (k in DRUG_EDGES and h in D) \
                or (k == "ASSOCIATED_WITH" and t in C):
            kept.append((h, k, t))
    proteins = {t for h, k, t in kept if k == "TARGETS"} | {h for h, k, t in kept if k == "ASSOCIATED_WITH"}
    kept += [(h, k, t) for h, k, t in triples if k in PROTEIN_EDGES and h in proteins]
    kept_set = set(kept)
    kept = [x for x in map(tuple, triples) if x in kept_set]          # original order
    used = {x for h, _, t in kept for x in (h, t)}
    sample_nodes = {kind: [n for n in lst if n[0] in used] for kind, lst in nodes.items()}

    # 3. files of the sample (original order everywhere; see sample/README.md)
    sample_drugs = {d: {"name": drugs[d]["name"], "treats": treats[d]} for d in drugs if d in D}
    sample_conds = {c: info for c, info in conditions.items() if c in C}
    sample_trials = {}
    for nct, v in trials.items():
        ds = [d for d in v["drugs"] if d in D]
        cs = [c for c in v["conditions"] if c in C]
        if ds and cs:
            sample_trials[nct] = {"nct_id": v.get("nct_id", nct), "phase": v["phase"],
                                  "conditions": cs, "drugs": ds}

    save(sample_drugs, os.path.join(a.out, "drugs_with_conditions_final.json"))
    save(sample_conds, os.path.join(a.out, "conditions.json"))
    save(sample_nodes, os.path.join(a.out, "graph_nodes.json"))
    save(sample_trials, os.path.join(a.out, "clinical_trials.json"))
    with open(os.path.join(a.out, "graph_triples.txt"), "w", encoding="utf-8") as f:
        f.writelines(f"{h} {k} {t}\n" for h, k, t in kept)

    kinds = defaultdict(int)
    for _, k, _ in kept:
        kinds[k] += 1
    print(f"ATC group {a.atc}: {len(sample_drugs)} drugs, {len(sample_conds)} conditions, "
          f"{len(pairs)} indications")
    print(f"graph: {sum(map(len, sample_nodes.values()))} nodes, {len(kept)} edges")
    for kind, lst in sample_nodes.items():
        print(f"  {kind:22s} {len(lst):6d} nodes")
    for k in sorted(kinds, key=kinds.get, reverse=True):
        print(f"  {k:26s} {kinds[k]:6d} edges")
    print(f"clinical trials: {len(sample_trials)}")


if __name__ == "__main__":
    main()
