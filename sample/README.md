# HeCoReS-DR sample: the whole pipeline in five minutes

This folder holds a **real subset** of the HeCoReS-DR data: the drugs of the cardiovascular system (ATC group C) with their indications, the part of the knowledge graph around them, and the clinical trials that involve them. With it, the whole pipeline, from building the graph to the result tables, runs in about five minutes on a laptop, without the full data (tens of gigabytes), without a DrugBank licence and without Neo4j.

It is meant for **checking that the code works and reproduces**: anyone can run it and compare the result tables with the ones in `expected/`. Its numbers are not the paper's results: the paper uses all 1,179 drugs and 722 conditions, 200 permutations and 100 experiments (see "What the results mean" below).

| | |
|---|---|
| Run it | `bash sample/run_sample.sh` (from the HeCoReS-DR main folder) |
| Time | about 5 minutes on 2 CPU cores (plus building the Docker image the first time) |
| Needs | Docker, **or** Python 3.13 with `pip install -r requirements.txt` (then add `--local`) |
| Size | 1.8 MB in five files (`inputs/`) |
| Full data | Zenodo, [doi:10.5281/zenodo.17202171](https://doi.org/10.5281/zenodo.17202171) |

---

## 1. Running the sample

From the HeCoReS-DR main folder:

```bash
bash sample/run_sample.sh                 # Docker (sudo docker, as standalone/run_hecores.sh)
bash sample/run_sample.sh --no-sudo       # Docker without sudo (e.g. Docker Desktop on a Mac)
bash sample/run_sample.sh --local         # your own Python: pip install -r requirements.txt first
bash sample/run_sample.sh --jobs 4        # more CPU cores for the HCS step
bash sample/run_sample.sh --keep          # keep the files of an earlier run instead of starting again
```

The script:

1. copies the five files of `sample/inputs/` into `data/sample/` (ignored by git, so nothing produced by the run can be committed by accident);
2. runs steps 11–16 of the pipeline with `standalone/run_hecores.sh`, the same runner as for the full data:

   ```bash
   bash standalone/run_hecores.sh --work data/sample --from graph --n-experiments 5 --n-perms 20
   ```

3. compares the four summary tables with `sample/expected/` and prints, for each, `identical` or `DIFFERENT`.

A good run ends with:

```
Comparison with sample/expected/
  summary_ct12.csv: identical
  summary_ct34.csv: identical
  summary_main.csv: identical
  summary_mps.csv: identical
All four tables are identical to the expected output.
```

The script exits with code 0 when all four tables are identical, and 1 otherwise. Every run also writes a log in `standalone/logs/`, whose header gives the commit of the code that was run.

---

## 2. What the run does

| # | Step | What it does on the sample | Output in `data/sample/` |
|---|---|---|---|
| 11 | `graph` | builds the heterogeneous graph from the nodes and edges | `graph.json`, `metagraph.json`, `nodes.tsv` |
| 12 | `hcs` | Hetnet Connectivity Search: path scores (DWPC) for every drug–condition pair along 10 metapaths, against 20 degree-preserving permutations | `graph.hetmat/path-counts/dwpc_results/all_dwpc_results_with_p.tsv` |
| 13 | `matrices` | HCS_0 (known indications), HCS_1 (HCS_0 plus significant HCS pairs), three random controls, drug and condition similarities | `ratings_*.csv`, `drug_sims.csv`, `cond_sims.csv` |
| 14 | `splits` | train/test pairs for the four test sets: DrugBank indications (main), clinical trials phases I/II and III/IV, metapath removal (RQ3) | `eval_revision/train_test_pairs*.json` |
| 15 | `evaluate` | ALSWR, PMF, LogisticMF and HCS alone, 5 experiments each | `eval_revision/results_*.csv` |
| 16 | `summary` | mean and standard deviation of every metric | `eval_revision/summary_*.csv` |

Steps 1–10 (from the raw sources to the graph files) are not part of the sample: they need the DrugBank XML (licensed), the text-mining models and Neo4j. The sample starts from their output, as the full pipeline does when started from the intermediate files on Zenodo (`standalone/README.md`, "Starting from intermediate files").

The evaluation uses the corrections of the revision (`eval_revision/`, the default). The published behaviour can be run on the sample too, with `bash sample/run_sample.sh --paper`; its tables then go to `data/sample/eval_paper/` and are not compared with `expected/`.

What the run produced on 12 Nov 2026:

| | Sample | Full data |
|---|---|---|
| Graph | 6,496 nodes, 18,390 edges | 19,114 nodes, 78,375 edges |
| Drug–condition pairs with a path (all metapaths) | 48,417 | — |
| HCS_0 (known indications) | 570 | 4,478 |
| HCS_1 (HCS_0 + pairs with adjusted p ≤ 0.05) | 1,357 | — |
| Clinical-trial test pairs, phases I/II / III/IV | 403 / 678 | — |
| HCS permutations / experiments | 20 / 5 | 200 / 100 |

---

## 3. What the sample contains

### 3.1 How it was selected

The sample was made with `sample/make_sample.py` from the five files the pipeline has after step 8 (the same files as on Zenodo):

1. **Drugs:** every drug with at least one ATC therapeutic group starting with **C** (cardiovascular system), taken from the graph's `HAS_THERAPEUTIC_GROUP` edges: 144 drugs.
2. **Indications:** their indications in `drugs_with_conditions_final.json`, cut with **the same rule as step 8** (`prune()` in `src/kg/processing_for_neo4j.py`): every drug keeps at least 2 indications and every condition at least 2 drugs, repeated until nothing changes. Result: **133 drugs, 105 conditions, 570 indications**.
3. **Graph:** for those drugs, all their own edges (ATC groups, MeSH categories, protein targets) and their indications; for those conditions, all their protein associations; for every protein kept, all its Gene Ontology edges. Nodes are those that appear in a kept edge.
4. **Clinical trials:** every trial with at least one sample drug and one sample condition, with its drug and condition lists cut to the sample. This is the rule of the full file too, where every drug and condition of a trial is one of the 1,179 drugs and 722 conditions (the evaluation stops with an error on any other).

The order of drugs, conditions, nodes and edges is the order of the full files. `make_sample.py` is deterministic: run again on the same files, it gives byte-identical files.

### 3.2 The files

| File | Content | Kept from the full file | Source |
|---|---|---|---|
| `drugs_with_conditions_final.json` | 133 drugs | DrugBank ID, name, `treats` (the indications) | DrugBank 5.1.13 (IDs, names), indications text-mined with HunFlair2 |
| `conditions.json` | 105 conditions | all fields: name, definition, parents, synonyms, slim terms | CTD MEDIC vocabulary (MeSH) |
| `graph_nodes.json` | 6,496 nodes in 10 kinds | `[identifier, name]` | as below |
| `graph_triples.txt` | 18,390 edges | `source RELATION target`, one per line | as below |
| `clinical_trials.json` | 4,624 trials | NCT ID, phase, conditions, drugs | ClinicalTrials.gov |

Nodes and edges of the graph:

| Node kind | Nodes | Source | | Edge | Edges |
|---|---|---|---|---|---|
| Drug | 133 | DrugBank | | `HAS_PROCESS` | 7,916 |
| Condition | 105 | CTD MEDIC | | `HAS_FUNCTION` | 2,924 |
| Protein | 825 | UniProt (via DrugBank and CTD) | | `HAS_COMPONENT` | 2,248 |
| BiologicalProcess | 3,331 | Gene Ontology | | `HAS_CATEGORY` | 1,759 |
| MolecularFunction | 1,007 | Gene Ontology | | `ASSOCIATED_WITH` | 1,745 |
| CellularComponent | 456 | Gene Ontology | | `INDICATED_FOR` | 570 |
| CategoryDrug | 374 | MeSH (via DrugBank) | | `HAS_CHEMICAL_GROUP` | 380 |
| ChemicalGroup | 148 | WHO ATC | | `HAS_PHARMACOLOGICAL_GROUP` | 337 |
| PharmacologicalGroup | 80 | WHO ATC | | `TARGETS` | 260 |
| TherapeuticGroup | 37 | WHO ATC | | `HAS_THERAPEUTIC_GROUP` | 251 |

Examples of what is in it: statins (atorvastatin, pravastatin), sartans (valsartan, azilsartan), beta blockers (atenolol), calcium channel blockers (amlodipine), antiarrhythmics (amiodarone), diuretics (amiloride, bendroflumethiazide), PCSK9 inhibitors (alirocumab); conditions such as hypertension (essential, malignant, pulmonary, renal), heart failure and hypertriglyceridemia. About a third of the conditions have the slim term "Cardiovascular disease"; the others are those the same drugs are also indicated for.

### 3.3 Licences

The sample contains **no DrugBank text**: of the DrugBank fields of `drugs_with_conditions_final.json` (description, indication, pharmacodynamics, mechanism of action, toxicity, and so on), only the drug's ID and name are kept. The indication pairs, the drug–target edges and the ATC and category edges are a subset of what the published Zenodo record already contains (`hcs_0.csv`, `graph.json`). Condition data come from CTD, as `data/processed/CTD_diseases.csv` in this repository; trial data from ClinicalTrials.gov; Gene Ontology terms from GO; protein names from UniProt. Anyone reusing the sample outside this pipeline should follow the terms of those sources and cite them (see the main `README.md`).

---

## 4. What the results mean

**The sample shows that the code runs and reproduces; it does not reproduce the paper's numbers.** With 133 drugs instead of 1,179, 20 permutations instead of 200 and 5 experiments instead of 100, every number differs from the paper's. The direction of the main comparison is nevertheless visible, e.g. in `expected/summary_main.csv` (AUC, mean of 5 experiments):

| Model | HCS_0 | HCS_1 | Random controls (mean of 3) |
|---|---|---|---|
| ALSWR | 0.761 | 0.835 | 0.563 |
| LogisticMF | 0.744 | 0.821 | 0.593 |
| PMF | 0.549 | 0.683 | 0.507 |
| HCS alone | 0.823 | | |

With 20 permutations, the smallest possible p-value is 1/21 ≈ 0.048, so a pair can still pass the 0.05 threshold, but the p-values are coarse. Use the full data for any scientific conclusion.

**Expected messages.** In step 14, lines such as `Warning: Only found 84 negatives for disease 44` are normal on the sample: with 133 drugs, a few conditions have fewer than the 100 candidate negatives the full data has. Two metapaths (`Dh>CH<hDi>CO`, `Dt>PR<tDi>CO`) give a difference of exactly 0 in `summary_mps.csv`, because on the sample no pair depends on them alone: removing either leaves all 1,357 pairs of HCS_1 (the log of step 14 prints `without Dh>CH<hDi>CO: 1357 positive pairs`).

---

## 5. Reproducibility

Every random step uses a fixed seed, so the same code and package versions give the same tables. Checked on 12 Nov 2026 (code of branch `revision` plus this folder, Python 3.13 with the versions of `requirements.txt`, `--local`):

- two runs from scratch gave identical `summary_*.csv` files, and identical metrics in every `results_*.csv` (only the column `time_seconds`, the running time, differs);
- the results do not depend on the number of cores: `--jobs 4` and `--jobs 2` gave the same tables;
- a run took 4 min 25 s on 2 cores.

### If the tables differ

1. **Package versions.** Differences in the last digit can come from other versions of numpy, scipy or scikit-learn. Compare yours with `requirements.txt` (`bash standalone/run_hecores.sh --check` prints them; with `--local`, `pip list`). The Docker image installs exactly the versions in `requirements.txt`.
2. **Changed options.** `--n-experiments`, `--n-perms`, `--models`, `--sets` or `--paper` change the results; `expected/` holds the defaults only.
3. **Changed inputs.** `git status sample/` must show nothing; `git diff sample/inputs/` shows any change.
4. **Changed code.** The log's `Commit:` line gives the code that was run; `git status --short` shows uncommitted changes.
5. **Larger differences** (whole rows, other pair counts in the log) mean the code or the inputs differ. Compare the log with the one of a good run: the counts printed by steps 12–14 (pairs with a path, significant pairs, clinical-trial pairs) must be the ones in section 2.

---

## 6. Making the sample again, or another one

With the five files of the full data after step 8 (from Zenodo, or `data/work/` after a full run):

```bash
python3 sample/make_sample.py --full data/work --out sample/inputs           # this sample (ATC group C)
python3 sample/make_sample.py --full data/work --out data/sample_N --atc N   # e.g. nervous system
```

`--atc` takes the first letter of an ATC code (A, B, C, D, G, H, J, L, M, N, P, R, S, V). Group C was chosen because it is one of the larger groups after pruning (133 drugs, 105 conditions, 570 indications) and is biologically coherent. To run another sample, put it in its own folder and use the runner directly:

```bash
bash standalone/run_hecores.sh --work data/sample_N --from graph --n-experiments 5 --n-perms 20
```

After changing `sample/inputs/` or the code, update the expected tables with a run and a copy:

```bash
bash sample/run_sample.sh
cp data/sample/eval_revision/summary_*.csv sample/expected/
```

---

## 7. Files in this folder

| File | What it is |
|---|---|
| `README.md` | this guide |
| `run_sample.sh` | runs the pipeline on the sample and compares the results with `expected/` |
| `make_sample.py` | builds the sample from the full data |
| `inputs/` | the five starting files (section 3) |
| `expected/` | the four summary tables of a run with the defaults (12 Nov 2026) |
