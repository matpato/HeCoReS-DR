"""
Summary tables of the results written by scofi.py.

    python -m src.evaluation.summarize [--bootstrap 10000]

Reads results_*.csv in the current (work) folder and writes
  summary_main{suffix}.csv, summary_ct12.csv, summary_ct34.csv
  (summary_ct12_active.csv, summary_ct34_active.csv with HECORES_CT_STATUS=active)
      mean and standard deviation over experiments of each metric, per model
      and training dataset; random controls also averaged into HCS_RANDOM
      (Reviewer 1: report them as one row with SD)
  summary_mps.csv
      RQ3: NDCG@10 of HCS_1 minus NDCG@10 without each metapath, per model,
      with a bootstrap 95% confidence interval of the mean difference
      (the thesis used 10^6 resamples; --bootstrap sets the number)
"""
import argparse
import glob
import re

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--bootstrap", type=int, default=10000)
a = ap.parse_args()

METRICS = ["AUC", "MRR", "PrecisionK", "RecallK", "F1K", "MAP", "NDCGk", "HRk"]
pattern = re.compile(r"results_(?P<model>[A-Za-z]+)_(?P<n>\d+)(?P<suffix>.*)\.csv$")
groups = {}
for f in sorted(glob.glob("results_*.csv")):
    m = pattern.match(f)
    if not m:
        continue
    suffix = m["suffix"] or ""
    kind = "mps" if suffix == "_mps" else (suffix[1:] if suffix.startswith("_ct") else f"main{suffix}")
    df = pd.read_csv(f)
    df["model"] = m["model"]
    groups.setdefault(kind, []).append(df)

for kind, dfs in groups.items():
    df = pd.concat(dfs, ignore_index=True)
    cols = [f"{x}_avg" for x in METRICS if f"{x}_avg" in df]
    if kind != "mps":
        rnd = df[df.dataset.astype(str).str.startswith("HCS_RANDOM")]
        rnd = rnd.groupby(["model", "experiment"])[cols].mean().reset_index().assign(dataset="HCS_RANDOM (mean of 3)")
        df = pd.concat([df, rnd], ignore_index=True)
        t = df.groupby(["model", "dataset"])[cols].agg(["mean", "std"]).round(3)
        t.columns = [f"{c.replace('_avg', '')}_{s}" for c, s in t.columns]
        t.insert(0, "n_experiments", df.groupby(["model", "dataset"]).experiment.nunique())
        t.to_csv(f"summary_{kind}.csv")
        print(f"\n=== {kind} (mean over experiments) ===")
        print(t[[c for c in t.columns if c.endswith("_mean") or c == "n_experiments"]].to_string())
    else:
        rng = np.random.default_rng(0)
        rows = []
        for model, d in df.groupby("model"):
            w = d.pivot(index="experiment", columns="dataset", values="NDCGk_avg")
            for mp in [c for c in w.columns if c not in ("HCS_0", "HCS_1")]:
                diff = (w["HCS_1"] - w[mp]).dropna().values
                boot = rng.choice(diff, size=(a.bootstrap, len(diff)), replace=True).mean(axis=1)
                rows.append({"model": model, "metapath_removed": mp, "n_experiments": len(diff),
                             "NDCG10_HCS1_minus_without": round(diff.mean(), 4),
                             "ci95_low": round(np.quantile(boot, 0.025), 4),
                             "ci95_high": round(np.quantile(boot, 0.975), 4)})
        t = pd.DataFrame(rows)
        t.to_csv("summary_mps.csv", index=False)
        print("\n=== RQ3: NDCG@10 of HCS_1 minus HCS_1 without one metapath ===")
        print(t.to_string(index=False))
