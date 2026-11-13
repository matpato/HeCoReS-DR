"""
Sensitivity analysis of the clinical-trial test sets: all trials vs. trials
that were not withdrawn, terminated or suspended (HECORES_CT_STATUS=active).

    python -m src.evaluation.compare_ct_status [--bootstrap 10000]

Run in the evaluation folder (e.g. data/work/eval_revision) after both runs.
Reads results_{model}_{n}_ct12.csv / _ct12_active.csv (and ct34) and writes
summary_ct_status.csv: per model and test set, mean NDCG@10 and AUC of HCS_0
and HCS_1, and the paired difference HCS_1 - HCS_0 with a bootstrap 95% CI.
"""
import argparse
import glob
import re

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--bootstrap", type=int, default=10000)
a = ap.parse_args()

pattern = re.compile(r"results_(?P<model>[A-Za-z]+)_(?P<n>\d+)_(?P<set>ct12|ct34)(?P<active>_active)?\.csv$")
rng = np.random.default_rng(0)
rows = []
for f in sorted(glob.glob("results_*_ct*.csv")):
    m = pattern.match(f)
    if not m or m["model"] == "HCS":
        continue
    df = pd.read_csv(f)
    w = df.pivot(index="experiment", columns="dataset", values="NDCGk_avg")
    auc = df.pivot(index="experiment", columns="dataset", values="AUC_avg")
    diff = (w["HCS_1"] - w["HCS_0"]).dropna().values
    boot = rng.choice(diff, size=(a.bootstrap, len(diff)), replace=True).mean(axis=1)
    rows.append({
        "test_set": m["set"].upper().replace("CT12", "CT I/II").replace("CT34", "CT III/IV"),
        "trials": "active only" if m["active"] else "all",
        "model": m["model"],
        "n_experiments": len(diff),
        "NDCG10_HCS0": round(w["HCS_0"].mean(), 3),
        "NDCG10_HCS1": round(w["HCS_1"].mean(), 3),
        "diff": round(diff.mean(), 3),
        "ci95_low": round(np.quantile(boot, 0.025), 3),
        "ci95_high": round(np.quantile(boot, 0.975), 3),
        "AUC_HCS0": round(auc["HCS_0"].mean(), 3),
        "AUC_HCS1": round(auc["HCS_1"].mean(), 3),
    })

if not rows:
    raise SystemExit("No results_*_ct12*.csv / *_ct34*.csv files found in this folder.")
t = pd.DataFrame(rows).sort_values(["test_set", "model", "trials"])
t.to_csv("summary_ct_status.csv", index=False)
print("NDCG@10 on the clinical-trial test sets: HCS_1 - HCS_0 (paired, bootstrap 95% CI)")
print(t.to_string(index=False))
