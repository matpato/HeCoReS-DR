# Running the HeCoReS-DR pipeline

`run_hecores.sh` runs every step of the pipeline, each in its own Docker container (or with your own Python with `--local`). Run every command from the HeCoReS-DR main folder.

## Requirements

- Docker (Docker Engine on Linux, Docker Desktop on Windows/macOS) with the buildx plugin, **or** Python 3.13 with `pip install -r requirements.txt`
- The runner calls `sudo docker`, as PharmaLinKG does (the server needs sudo for Docker); on a computer where Docker runs without sudo (e.g. Docker Desktop on a Mac) add `--no-sudo`
- Internet: to build the image, for MeSH (step 3), the HunFlair2 models (step 4), BNNR's Octave code
- 2 CPU cores and 8 GB RAM; about 5 GB disk (step 12 writes 200 permuted graphs)
- Step 4 (text mining) uses a larger image with torch (`hecores-dr:ner`, CPU); a GPU makes it much faster
- Step 9 starts a Neo4j container (`neo4j:5`); with `--local`, a Neo4j server at `NEO4J_URI`

## Steps

```
 1 ctd → 2 drugbank → 3 mesh → 4 conditions → 5 atc → 6 proteins → 7 go → 8 filter → 9 neo4j → 10 trials
 → 11 graph → 12 hcs → 13 matrices → 14 splits → 15 evaluate → 16 summary
```

| # | Step | Script | Reads | Writes |
|---|---|---|---|---|
| 1 | `ctd` | `src/preprocessing/ctd_processing.py` | raw `CTD_diseases.csv` | `conditions.json`, `conditions_no_slims.json` |
| 2 | `drugbank` | `src/preprocessing/drugbank_processing.py` | raw `full database.xml` | `drugs.json`, `targets.json`, `proteins.json`, `drug_categories.json` |
| 3 | `mesh` | `src/preprocessing/mesh.py` | MeSH 2025 (download) | `mesh.json` |
| 4 | `conditions` | `src/preprocessing/extract_conditions.py` | `drugs.json`, `conditions.json` | `drugs_with_conditions.json` |
| 5 | `atc` | `src/preprocessing/atc_codes_processing.py` | raw `WHO ATC-DDD 2024-07-31.csv` | `atc.json` |
| 6 | `proteins` | `src/kg/condition_associations.py` | raw `CTD_genes.csv`, `CTD_curated_genes_diseases.csv` | `proteins.json` |
| 7 | `go` | `src/preprocessing/go_target_processing.py` | raw `go.obo` | `go.json`, `protein_map.json`, `targets.json` |
| 8 | `filter` | `src/kg/processing_for_neo4j.py` | step 4 output | `drugs_with_conditions_final.json`, `conditions.json` |
| 9 | `neo4j` | `src/kg/neo4j.py` | steps 2–8 | `graph_triples.txt`, `graph_nodes.json` |
| 10 | `trials` | `src/preprocessing/clinical_trials.py` | raw `ctg-public-xml.zip` | `clinical_trials.json` |
| 11 | `graph` | `src/models/kg_enhanced/hpy.py` | step 9 output | `graph.json`, ablation graphs |
| 12 | `hcs` | `src/models/kg_enhanced/hmp.py` | `graph.json` | `graph.hetmat/.../all_dwpc_results_with_p.tsv` (~45 min) |
| 13 | `matrices` | `src/models/similarity_based/sims.py` | steps 8, 12 | `ratings_hcs0.csv`, `ratings_hcs1.csv`, random controls, similarities |
| 14 | `splits` | `src/models/training_testing.py` | step 13 | `train_test_pairs*.json` |
| 15 | `evaluate` | `src/evaluation/scofi.py` | step 14 | `results_*.csv` |
| 16 | `summary` | `src/evaluation/summarize.py` | step 15 | `summary_*.csv` |

A step whose input files are missing is **skipped** with a message. `--status` shows which steps are ready.

## Folders

| Folder | Holds |
|---|---|
| `data/raw/` | downloaded source files (see the table above) |
| `data/work/` | intermediate files; steps 1–13 run here |
| `data/work/eval_revision/` | steps 14–16 with the corrections of the revision (default) |
| `data/work/eval_paper/` | steps 14–16 as published (`--paper`) |
| `standalone/logs/` | a log of every run |

## Starting from intermediate files

Without the DrugBank XML (licensed), start from the files the student kept. Put them in `data/work/`:

| File | Rename to |
|---|---|
| `drugs_with_conditions.json` (text-mining output, step 4) | – |
| `conditions.json` (all 13,299 MEDIC conditions, step 1) | – |
| `graph_nodes3.json` | `graph_nodes.json` |
| `graph_triples.txt` | – |
| `clinical_trials.json` | – |

Then `bash standalone/run_hecores.sh` runs steps 8 and 11–16 and skips the others. Checked on 9 Nov 2026: step 8 gives the 1,179 drugs, 722 conditions and 4,478 indications of the released `hcs_0.csv`; step 11 gives the released `graph.json`; step 13 gives the released `drug_sims.csv` and `disease_sims.csv`.

## Commands

```bash
bash standalone/run_hecores.sh --check              # build the image, check the packages
bash standalone/run_hecores.sh --status             # which steps can run
bash standalone/run_hecores.sh --quick              # installation test (about 1 h), in data/quick; compare with standalone/expected/quick/
bash standalone/run_hecores.sh                      # all steps (about 6-8 h from step 8 on 2 cores)
bash standalone/run_hecores.sh --step hcs           # one step (name or number)
bash standalone/run_hecores.sh --from matrices      # from a step to the end
bash standalone/run_hecores.sh --paper --from splits    # evaluation as published (reproduces the paper)
bash standalone/run_hecores.sh --pct "20 40 60 80" --from hcs   # ablation (RQ2)
bash standalone/run_hecores.sh --models "ALSWR HCS" --sets "main"   # some models / test sets
bash standalone/run_hecores.sh --n-experiments 50   # fewer experiments
bash standalone/run_hecores.sh --no-sudo            # plain docker instead of sudo docker (e.g. a Mac)
bash standalone/run_hecores.sh --local              # without Docker
bash standalone/run_hecores.sh --shell 'ls -la'     # a command in the container (in data/work)
bash standalone/run_hecores.sh --rebuild --check    # rebuild the image (after editing requirements*.txt)
bash standalone/run_hecores.sh --dry-run            # print the commands only
```

Steps 12, 14 and 15 keep their work when interrupted: rerun the same command and they continue.

## Settings

All scripts read their settings from environment variables (`utils/config.py`); the runner sets them from its options.

| Variable | Default | Meaning |
|---|---|---|
| `HECORES_RAW` | `data/raw` | folder of the source files |
| `HECORES_PCT` | (empty) | ablation stage: % of edges removed |
| `HECORES_N_EXPERIMENTS` | 100 | experiments per model (BNNR ≤ 5, RQ3 ≤ 50) |
| `HECORES_MODELS` | `ALSWR,PMF,LogisticMF,HCS` | models of step 15 |
| `HECORES_N_PERMS` / `HECORES_JOBS` | 200 / 2 | HCS permutations / CPU cores |
| `HECORES_METRICS` | `scores` | `labels` = published (metrics on thresholded labels) |
| `HECORES_NEGATIVES` | `independent` | `paper` = published (negatives exclude HCS and random positives) |
| `HECORES_RQ3` | `corrected` | `paper` = published RQ3 matrices |
| `HECORES_PAPER_MODE` | 0 | 1 = all three as published (`--paper`) |
| `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` | container `neo4j` | Neo4j connection |

What changed from the published code, and why: [`../CHANGES_REVISION.md`](../CHANGES_REVISION.md).
