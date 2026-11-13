"""
Evaluation of the collaborative-filtering models on the precomputed
train/test pairs (src/models/training_testing.py).

    python -m src.evaluation.scofi [--sets main ct12 ct34 mps] [--models ALSWR PMF LogisticMF HCS BNNR]

main   DrugBank test set (20% of HCS_0), models trained on HCS_0, HCS_1 and the
       random controls -- Table 4 (full graph) or one ablation stage (HECORES_PCT)
ct12   clinical-trial test set, phases I/II   (full graph only)
ct34   clinical-trial test set, phases III/IV (full graph only)
mps    RQ3: HCS_1 and HCS_1 without each metapath (full graph only)

Models: HECORES_MODELS or --models. HCS = HCS alone (no collaborative
filtering), the baseline added for the revision. BNNR needs Octave and runs
HECORES_N_EXPERIMENTS capped at 5 experiments, as in the paper.

Output: results_{model}_{n}{suffix}.csv (main), results_{model}_{n}_ct12.csv,
results_{model}_{n}_ct34.csv, results_{model}_{n}_mps.csv in the work folder,
one row per experiment and training dataset; written after every experiment,
and an interrupted run continues where it stopped (rows already present are kept).

Metrics: on the model scores (HECORES_METRICS=scores, revision) or on the
thresholded labels (HECORES_METRICS=labels, published); see validation.py.
"""
import csv
import os
import random
from time import time

import numpy as np
import pandas as pd

import src.evaluation.validation as validation
import src.models.models as models
from src.models.training_testing import (build_dataset_from_pairs, ct_pair_sets, create_coo_matrix,
                                         load_datasets, load_precomputed_pairs, n_metapaths,
                                         reconstruct_datasets_from_pairs, unique_metapath_pairs)
from utils.config import (PCT, SFX, HETMAT_DIR, MODELS, N_EXPERIMENTS, N_EXPERIMENTS_BNNR,
                          N_EXPERIMENTS_MPS, RQ3, CT_STATUS, CT_SFX, CT_STOPPED, describe)


def run_pipeline(model_class, reconstructed_sets, params=None, n_experiments=10, output_file="results.csv"):
    """
    Run experiments using precomputed and reconstructed train/test datasets.

    Parameters:
        model_class: a class (not instance) that implements .fit() and .predict_proba()
        reconstructed_sets: output from `reconstruct_datasets_from_pairs(...)`
        params: parameters passed to the model
        n_experiments: number of experiments to run
        output_file: CSV written after every experiment
    """
    results, done = [], set()
    if os.path.exists(output_file):
        old = pd.read_csv(output_file)
        old = old[old["experiment"] <= n_experiments]
        results = old.to_dict("records")
        done = set(old["experiment"])
        if done:
            print(f"{output_file}: experiments {sorted(done)[:3]}... already done, continuing")
    print(f"========== {model_class.__name__} -> {output_file} ==========")

    for i in range(n_experiments):
        if i + 1 in done:
            continue
        print(f"\n========== EXPERIMENT {i+1} ==========")
        random.seed(i)
        np.random.seed(i)

        test_dataset = reconstructed_sets[i]['test_dataset']
        train_datasets = reconstructed_sets[i]['train_datasets']

        for name, train_dataset in train_datasets.items():
            if model_class is models.HCS and results and any(r["experiment"] == i + 1 for r in results):
                break                     # HCS does not depend on the training dataset: once per experiment
            print(f"--------------- {name.upper()} ---------------")
            start = time()
            model = model_class(params)
            model.fit(train_dataset, seed=i)
            scores = model.predict_proba(test_dataset)
            predictions = model.predict(scores)
            metrics, _ = validation.compute_metrics(
                scores, predictions, test_dataset,
                metrics=validation.metrics_list, k=10, beta=1, verbose=False
            )
            row = {'experiment': i + 1,
                   'dataset': name if model_class is not models.HCS else "-",
                   'time_seconds': time() - start}
            for metric_name, values in metrics.iterrows():
                row[f'{metric_name}_avg'] = values['Average']
                row[f'{metric_name}_std'] = values['StandardDeviation']
            results.append(row)

        with open(output_file, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    print(f"\nResults saved to {output_file}")


def main(sets, model_names):
    print(describe())
    d, users, items = load_datasets()
    hcs_0 = d["HCS_0"]
    shape = hcs_0.ratings.shape
    lookup = dict(d)
    n = N_EXPERIMENTS

    jobs = []   # (set name, main dataset name, pairs file, n experiments, output suffix, datasets)
    if "main" in sets:
        jobs.append(("main", "HCS_0", f"train_test_pairs{SFX}{n}.json", n, SFX, lookup))
    if not PCT and ({"ct12", "ct34"} & set(sets)):
        ct_pairs12, ct_pairs34 = ct_pair_sets('clinical_trials.json', "drugs_with_conditions_final.json",
                                              exclude_status=CT_STOPPED if CT_STATUS == "active" else None)
        for name, pairs in (("CT12", ct_pairs12), ("CT34", ct_pairs34)):
            if name.lower() in sets:
                ct_lookup = {**lookup, name: build_dataset_from_pairs(list(pairs), users, items, name,
                                                                      same_item_user_features=False)}
                jobs.append((name.lower(), name, f"train_test_pairs_{name.lower()}{CT_SFX}_{n}.json", n,
                             f"_{name.lower()}{CT_SFX}", ct_lookup))
    if not PCT and "mps" in sets:
        f = f'{HETMAT_DIR}/path-counts/dwpc_results/all_dwpc_results_with_p.tsv'
        hcs_1 = d["HCS_1"]
        hcs1_pairs = {(hcs_1.user_list[u], hcs_1.item_list[i]) for i, u in hcs_1.get_positive_indices()}
        known = {(hcs_0.user_list[u], hcs_0.item_list[i]) for i, u in hcs_0.get_positive_indices()}
        mps_lookup = {"HCS_0": hcs_0, "HCS_1": hcs_1}
        for mp in n_metapaths:
            pairs = unique_metapath_pairs(f, mp, base_pairs=None if RQ3 == "paper" else hcs1_pairs, known_pairs=known)
            mps_lookup[mp] = build_dataset_from_pairs(list(pairs), users, items, mp, same_item_user_features=False)
        jobs.append(("mps", "HCS_0", f"train_test_pairs_mps_{N_EXPERIMENTS_MPS}.json", N_EXPERIMENTS_MPS,
                     "_mps", mps_lookup))

    for set_name, main_name, pairs_file, n_exp, suffix, lookup in jobs:
        if not os.path.exists(pairs_file):
            print(f"{pairs_file} not found (run training_testing.py first): '{set_name}' skipped")
            continue
        saved = load_precomputed_pairs(pairs_file)
        n_exp = min(n_exp, len(saved))
        sets_ = reconstruct_datasets_from_pairs(main_name, saved, dataset_lookup=lookup, shape=shape,
                                                create_coo_matrix=create_coo_matrix)
        for model_name in model_names:
            model_class = getattr(models, model_name)
            if set_name == "mps" and model_name in ("BNNR", "HCS"):
                continue              # RQ3 compares training sets; BNNR excluded as in the paper
            n_model = min(n_exp, N_EXPERIMENTS_BNNR) if model_name == "BNNR" else n_exp
            run_pipeline(model_class, reconstructed_sets=sets_, n_experiments=n_model,
                         output_file=f"results_{model_name}_{n_model}{suffix}.csv")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Evaluate the models (see the docstring).")
    ap.add_argument("--sets", nargs="+", default=["main", "ct12", "ct34"],
                    choices=["main", "ct12", "ct34", "mps"])
    ap.add_argument("--models", nargs="+", default=MODELS)
    a = ap.parse_args()
    main(a.sets, a.models)
