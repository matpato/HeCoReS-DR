# Running HeCoReS-DR on the server — step by step

Written for the revision of BIOINF-2026-1006 (November 2026). Tested on the lab
server on 9 Nov 2026 (Docker with `sudo`, 8 cores). Every command is run from the
HeCoReS-DR folder (`cd ~/HeCoReS-DR`) unless said otherwise.

## 0. What you need

| | |
|---|---|
| Code | HeCoReS-DR, branch `revision` |
| Docker | with the buildx plugin; the runner calls `sudo docker` (on a Mac with Docker Desktop add `--no-sudo`) |
| Input files | the student's files, in `data/work/` (step 2 below) |
| Not needed | DrugBank account or XML, Neo4j, CTD/ATC/GO files, ClinicalTrials.gov archive (they only rebuild the knowledge graph from scratch, steps 1–7, 9, 10) |

## 1. Get the code (once)

```bash
git clone https://github.com/matpato/HeCoReS-DR.git
cd HeCoReS-DR
git fetch /path/to/HeCoReS-DR-revision.bundle revision:revision
git checkout revision
git am /path/to/0001-*.patch /path/to/0002-*.patch /path/to/0003-*.patch /path/to/0004-*.patch
git log --oneline -1          # must show the last patch
```

(If the branch is already on GitHub: `git checkout revision` after cloning is enough.)

## 2. Put the input files in place (once)

```bash
mkdir -p data/work data/raw
cp drugs_with_conditions.json conditions.json graph_triples.txt clinical_trials.json data/work/
cp graph_nodes3.json data/work/graph_nodes.json        # new name
```

Leave `data/raw/` **empty**. If `CTD_diseases.csv` is there, step 1 runs and
overwrites the student's `conditions.json`.

## 3. Check the installation (≈ 1 h)

```bash
bash standalone/run_hecores.sh --check       # builds the image, lists package versions
bash standalone/run_hecores.sh --status      # "filter" and "graph" must say ready
bash standalone/run_hecores.sh --quick       # whole pipeline, 2 experiments, in data/quick
```

`--status`: the later steps show "missing" because their inputs are written
during the run; steps 1–7, 9, 10 stay "missing" and are skipped. That is expected.

The quick test is fine when `data/quick/eval_revision/` holds
`summary_main.csv`, `summary_ct12.csv`, `summary_ct34.csv`, `summary_mps.csv`.
Its numbers mean nothing (2 permutations, 2 experiments). Reference (9 Nov 2026):
the quick-test RQ3 table was identical on the server and on the test machine,
e.g. ALS-WR `Di>CO<aPRa>CO` 0.0132 [0.0106, 0.0157].

What the log of step 8 (filter) must show — the published numbers:

| Log line | Expected |
|---|---|
| `17430` then `17403` | 17,430 DrugBank drugs, minus 27 without context (thesis Table A.2) |
| `There are 4450 approved drugs.` | 4,477 approved minus those 27 |
| `1515 conditions associated with drugs.` | thesis: 1,515 |
| `134` | excluded condition names (128 of thesis Table A.4 + 6 without Slim terms) |
| `78375` and the node table (Condition 722, Drug 1179, …) | Tables 1–2 of the paper |

## 4. tmux in five keys

tmux keeps programs running on the server after you close the terminal or the
connection drops. Each key is **two moves**: hold Control, press b, **release
both**, then press the key alone. (Holding Control while pressing c sends
Control-c, which stops the running program.)

| | |
|---|---|
| start a session | `tmux new -s hecores` (a green bar appears at the bottom) |
| new window / go to window 1 / next window | Control-b `c` / Control-b `1` / Control-b `n` |
| scroll up | Control-b `[`, arrows or PgUp, `q` to stop |
| leave everything running | Control-b `d` |
| come back (after `ssh`) | `tmux attach -t hecores` |

If `tmux new -s hecores` says the session exists, use `tmux attach -t hecores`.
Pressing arrow keys while a program runs prints `^[[A` / `^[[B`: harmless.

## 5. The full run (corrected version, as in the revised paper)

### 5a. Shared steps (≈ 30–40 min) — tmux window 0

```bash
tmux new -s hecores
cd ~/HeCoReS-DR
for s in filter graph hcs matrices; do bash standalone/run_hecores.sh --step $s --jobs 8 || break; done
```

Step `hcs` prints nothing for ~18 min while it builds 200 permuted graphs
(`200 permutations built in 1105 s` on the server), then a progress bar
`Null distributions … /200` (~14 min with 8 cores), then `Path scores` (~4 min).
`sudo docker stats --no-stream` (in another window) shows the container working.

### 5b. Evaluation, three test sets in parallel (several hours)

From a **second terminal tab** (⌘T on the Mac, `ssh` to the server), paste:

```bash
cd ~/HeCoReS-DR
R=standalone/run_hecores.sh
W='echo "waiting for step matrices..."; until [ -f data/work/ratings_hcs_random3.csv ]; do sleep 60; done; sleep 30'

tmux new-window -t hecores -n main   "cd ~/HeCoReS-DR; $W; bash $R --step splits --sets main && bash $R --step evaluate --sets main; exec bash"
tmux new-window -t hecores -n trials "cd ~/HeCoReS-DR; $W; bash $R --step splits --sets 'ct12 ct34' && (bash $R --step evaluate --sets ct12 & bash $R --step evaluate --sets ct34 & wait); exec bash"
tmux new-window -t hecores -n rq3    "cd ~/HeCoReS-DR; $W; bash $R --step splits --sets mps && bash $R --step evaluate --sets mps; exec bash"
```

The bar then shows `0:bash 1:main 2:trials 3:rq3`. Windows 1–3 stay quiet
("waiting for step matrices...") until window 0 has finished, then start by
themselves. When they start, look at each (Control-b `1`, `2`, `3`): if one asks
`[sudo] password`, type it there. Then Control-b `d` and log out.

You can also do it one after another in a single window (simpler, over a day):
`bash standalone/run_hecores.sh --from splits`.

### 5c. Summary tables (seconds)

When windows 1–3 are back at the prompt:

```bash
bash standalone/run_hecores.sh --step summary
ls data/work/eval_revision/summary_*.csv
```

## 6. Published behaviour, for the response letter

Same as 5b and 5c with `--paper` added to every `$R` command (results in
`data/work/eval_paper/`, the corrected results are not touched). 5a is not
repeated: the graph, HCS and matrices are the same.

```bash
bash standalone/run_hecores.sh --paper --from splits          # one window, or the 3 windows of 5b with --paper
```

## 7. Ablation (RQ2)

```bash
bash standalone/run_hecores.sh --pct "20 40 60 80" --from hcs --jobs 8
```

Four times step 12 (HCS on the pruned graph), then matrices, splits and
evaluation on the DrugBank test set for each. Results: `summary_main_20pct.csv`, …

## 8. What to send back

```
data/work/eval_revision/summary_*.csv
data/work/eval_paper/summary_*.csv
data/work/graph.hetmat/path-counts/dwpc_results/all_dwpc_results_with_p.tsv   (large; HCS p-values)
standalone/logs/        (the logs of the full run)
```

## 9. Interrupted runs

Every long step continues where it stopped: rerun the same command.
`hcs` keeps the permuted graphs; `splits` and `evaluate` keep finished experiments.

## 10. Troubleshooting (all seen on 9 Nov 2026)

| Message | Cause | Fix |
|---|---|---|
| `unknown shorthand flag: 't' in -t` | Docker's buildx plugin is missing | `sudo apt-get install docker-buildx-plugin` (the runner now says so) |
| `~/.docker/config.json: permission denied` | Docker was once run with sudo | harmless with sudo; or `sudo chown -R $USER:$(id -gn) ~/.docker` |
| Safari: "the address is invalid" on `bolt://neo4j:7687` | a setting in the log, shown as a link | nothing to do (only step 9 uses Neo4j) |
| Step 16 printed its command but no tables yet | it was still running when looked at | `bash standalone/run_hecores.sh --quick --step summary` |
| tmux: no new window with Control-b `c` | Control still held when pressing c | release both keys after b; or use `tmux new-window` from another tab (5b) |
| `tmux new -s hecores`: duplicate session | already created | `tmux attach -t hecores` |
| DrugBank login? | not needed | only step 2 (rebuilding from scratch) needs the DrugBank XML; never put the password in a file |

## 11. Times measured on the server (9 Nov 2026, 8 cores)

| Step | Time |
|---|---|
| `--check` (first build) | a few minutes |
| `--quick` | ≈ 55 min (evaluation ≈ 16 min per experiment for the four test sets together) |
| `hcs`, full | ≈ 18 min permutations + ≈ 14 min null distributions + ≈ 4 min path scores |
| `splits` + `evaluate`, 100 experiments | estimate: ≈ 8–10 h with the three windows of 5b; over a day in one window |
