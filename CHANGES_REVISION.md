# Changes for the revision (November 2026)

Branch `revision`, starting from commit `8b31f9d` (the code archived with the submitted manuscript BIOINF-2026-1006). Each change is a separate commit; `git log --oneline 8b31f9d..revision` lists them. The tables name each commit by the start of its title (commit hashes change when the commits are applied with `git am`).

These changes were prepared with the assistance of an AI assistant (see the disclaimer in [`README.md`](README.md#disclaimer-use-of-ai-assistance)).

## 1. Fixes that make the published code run

| Problem in the published code | Change | Commit |
|---|---|---|
| Source files read from fixed absolute paths (`/data/processed/...`, `/workspace/hcs/...`) | read from `HECORES_RAW`; intermediate files stay in the current folder | "Read source files from a configurable folder" |
| Neo4j password written in the code (`password`; the README says `hecores_password`) | from `NEO4J_*` settings | "neo4j.py: connection from settings" |
| Neo4j load used MERGE on a non-empty database, so an earlier load was exported with the new graph; errors were printed and ignored | the load empties the database first (`NEO4J_KEEP=1` keeps it); errors stop the script | "neo4j.py: connection from settings" |
| `hpy.py` scanned all nodes for every edge (~1.5×10⁹ comparisons) | dictionary lookup; same `graph.json` | "hpy.py: look up node kinds" |
| `hmp.py` stopped with `NameError` (`collections`), wrote ~130 MB × 200 permuted results to disk and re-read a Parquet file per (permutation, degree group) | same statistics, computed in memory (~45 min on 2 cores) | "hmp.py: rewrite the p-value computation" |
| Ablation stage chosen by editing `pct = 20` / `pct = 80` and file names by hand; `graph_triples_{pct}pct.txt` written by no script | `HECORES_PCT`; `hpy.py` writes the triples of each stage | "Ablation stage from HECORES_PCT" |
| `np.asfarray` removed in NumPy 2.0 | `np.asarray(..., dtype=float)` | "Rank metrics on model scores" |
| `Dataset` recognised DataFrames by their printed type name, which pandas ≥ 3 and recent scipy changed: IDs and feature order were silently lost | `isinstance` | "datasets.py: recognise DataFrames" |
| `is_positive_pair` converted the full matrix to dense on every call | converted once | "datasets.py: is_positive_pair()" |
| `training_testing.py` and `scofi.py` ran everything on import; `scofi.py` imported the undefined `unrelated_pairs`; settings edited by hand | `main()` with command-line options | "training_testing.py: run as a script", "scofi.py: run as a script" |
| BNNR downloaded its Octave code from the `master` branch at run time | the commit named in the code (`68f8e98c`) | "BNNR: download BNNR.m and svt.m" |
| `processing_for_neo4j.py` needed the DrugBank-derived protein files even for the drug–condition part | that part runs without them | "processing_for_neo4j.py" |
| `requirements.txt` listed `python>=3.8` (not a package), unused packages, and lacked hetnetpy, hetmatpy, pronto, flair, neo4j | pinned, tested versions; text mining in `requirements-ner.txt` | "Add the pipeline runner" |
| `docker/Dockerfile` copied `data/` and `experiments/`, which do not exist | `standalone/Dockerfile` and `standalone/run_hecores.sh` | "Add the pipeline runner" |
| Every `splits` call rebuilt the DrugBank splits, so evaluation windows run in parallel wrote the same file at once; split files written in place | each call builds only the requested test sets; split files written to a temporary file and renamed | "Splits: build only the requested test sets" |

## 2. Corrections that change the results

Each correction can be switched back to the published behaviour (`HECORES_PAPER_MODE=1`, or `run_hecores.sh --paper`), which reproduces the published numbers.

| | Published | Revision | Setting | Commit |
|---|---|---|---|---|
| **C1. Metrics** | AUC, MRR, P@10, R@10, F1@10, MAP, NDCG@10, HR@10 computed by ranking the thresholded +1/−1 predictions (score ≥ 0.5) | computed on the model scores (F-score still on labels) | `HECORES_METRICS` | "Rank metrics on model scores" |
| **C2. Test negatives** | a pair could be a test negative only if it was not positive in HCS_1 or in any random control, so pairs flagged by HCS were never negatives | negatives exclude only known positives of the test set's source (HCS_0; HCS_0 + trial pairs for the clinical-trial sets) | `HECORES_NEGATIVES` | "training_testing.py: run as a script" |
| **RQ3 matrices** | "HCS_1 without metapath k" = significant pairs minus those unique to k; lacked the 1,902 known indications that are not significant | HCS_1 minus the pairs unique to k, known indications kept (as described in the thesis) | `HECORES_RQ3` | "training_testing.py: run as a script" |
| **Path scores** | `percent_dwpc` divided a raw path weight by the transformed DWPC (did not sum to 100%) | divided by the raw DWPC | – | "hmp.py: rewrite the p-value computation" |

## 3. Additions

- `models.HCS`: ranks pairs by HCS alone, without collaborative filtering (Reviewer 1) — "Add HCS, a baseline"
- RQ3 is evaluated by `scofi.py --sets mps` (the published script built the sets but never ran them) — "scofi.py: run as a script"
- `src/evaluation/summarize.py`: mean ± SD tables, random controls averaged (Reviewer 1), RQ3 bootstrap CIs — "Add summarize.py"
- `standalone/`: Docker image and runner, as in MedJsonify — "Add the pipeline runner"
- `sample/`: a real subset of the data (133 cardiovascular drugs, 105 conditions, 570 indications, 4,624 trials) that runs steps 11–16 in about 5 minutes and is compared with expected tables; `run_hecores.sh --n-perms` sets the number of HCS permutations (Reviewers) — "Add a runnable sample of the data"

## 4. Checks (9–11 November 2026)

Starting from the student's `drugs_with_conditions.json`, `conditions.json`, `graph_nodes3.json` and `graph_triples.txt`:

| Output | Compared with | Result |
|---|---|---|
| step 8: drugs, conditions, indications | released `hcs_0.csv` | identical (1,179 × 722, 4,478) |
| step 11: `graph.json` | student's `graph.json` | identical nodes and edges, same order (19,114 / 78,375) |
| step 13: similarity matrices, HCS_0 | released `drug_sims.csv`, `disease_sims.csv`, `hcs_0.csv` | identical |
| step 12: pairs added by HCS | released `hcs_1.csv` (28,054 added pairs) | 27,512 reproduced, 0 extra; the other 542 have adjusted p 0.050–0.069 here (533 for `Dt>PRh>M<hPRa>CO`): the student's null distribution for that metapath was slightly less conservative in some degree groups; the cause cannot be checked without his intermediate files |
| step 15 (HCS_0, experiments 1–2) | independent re-implementation (audit) | equal to 4 decimals for ALS-WR, BPR, LMF |
| sample (`sample/run_sample.sh`, 12 Nov 2026) | a second run from scratch, with another number of cores | identical tables and per-experiment metrics; `--local`, Python 3.13 |
| full run, steps 8 and 11–16, in Docker on the lab server (8 cores) | – | completed 9–10 Nov 2026: 100 experiments for ALS-WR, BPR, LMF and HCS on the DrugBank and both clinical-trial test sets, 50 for RQ3; the quick test gave the same numbers on the server and on the test machine |

## 5. Not tested here

- Steps 1–7, 9 and 10 (need the DrugBank XML, Neo4j, HunFlair2 models and the ClinicalTrials.gov archive).
- BNNR (needs Octave, image `hecores-dr:octave`); not part of the default models.
