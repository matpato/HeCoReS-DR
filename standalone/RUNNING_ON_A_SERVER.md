# Running HeCoReS-DR on a server

Long runs with Docker on a Linux server, measured on an 8-core server in November 2026. Every command is run from the HeCoReS-DR folder (`cd ~/HeCoReS-DR`). Getting the code, the input files, the steps and all options are in [`README.md`](README.md); the server needs `tmux`, which keeps a run going after the connection closes (`tmux new -s hecores` to start, Control-b then `d` to leave, `tmux attach -t hecores` to come back).

## 1. Check the installation (≈ 1 h)

```bash
bash standalone/run_hecores.sh --check       # builds the image, lists package versions
bash standalone/run_hecores.sh --status      # "filter" and "graph" must say ready
bash standalone/run_hecores.sh --quick       # whole pipeline, 2 experiments, in data/quick
```

`--status`: the later steps show "missing" because their inputs are written
during the run; steps 1–7, 9, 10 stay "missing" and are skipped. That is expected.

The quick test is fine when `data/quick/eval_revision/` holds
`summary_main.csv`, `summary_ct12.csv`, `summary_ct34.csv`, `summary_mps.csv`.
Its numbers mean nothing (2 permutations, 2 experiments), but they should
match the reference tables in `standalone/expected/quick/` (the README there
gives the `diff` commands). Reference (9 Nov 2026): the quick-test RQ3 table was
identical on the server and on the test machine, e.g. ALS-WR `Di>CO<aPRa>CO`
0.0132 [0.0106, 0.0157].

What the log of step 8 (filter) must show — the published numbers:

| Log line | Expected |
|---|---|
| `17430` then `17403` | 17,430 DrugBank drugs, minus 27 without context (D. Aveiro, MSc thesis, Table A.2) |
| `There are 4450 approved drugs.` | 4,477 approved minus those 27 |
| `1515 conditions associated with drugs.` | thesis: 1,515 |
| `134` | excluded condition names (128 of the thesis' Table A.4 + 6 without Slim terms) |
| `78375` and the node table (Condition 722, Drug 1179, …) | Tables 1–2 of the paper |

## 2. The full run (corrected evaluation, as in the revised article)

### 2a. Shared steps (≈ 30–40 min) — tmux window 0

```bash
tmux new -s hecores
cd ~/HeCoReS-DR
for s in filter graph hcs matrices; do bash standalone/run_hecores.sh --step $s --jobs 8 || break; done
```

Step `hcs` prints nothing for ~18 min while it builds 200 permuted graphs
(`200 permutations built in 1105 s` on 8 cores), then a progress bar
`Null distributions … /200` (~14 min with 8 cores), then `Path scores` (~4 min).
`sudo docker stats --no-stream` (in another window) shows the container working.

### 2b. Evaluation, three test sets in parallel (several hours)

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

### 2c. Summary tables (seconds)

When windows 1–3 are back at the prompt:

```bash
bash standalone/run_hecores.sh --step summary
ls data/work/eval_revision/summary_*.csv
```

Add `--paper` to the commands of 2b and 2c for the published behaviour
(results in `data/work/eval_paper/`; 2a is not repeated).

## 3. Ablation (RQ2)

```bash
bash standalone/run_hecores.sh --pct "20 40 60 80" --from hcs --jobs 8
```

Four times step 12 (HCS on the pruned graph), then matrices, splits and
evaluation on the DrugBank test set for each. Results: `summary_main_20pct.csv`, …

## 4. Interrupted runs

Rerun the same command. `hcs` keeps the permuted graphs and `evaluate` keeps
finished experiments, so both continue where they stopped; `splits` starts again
from the first experiment and gives the same splits (fixed random seeds).

## 5. Troubleshooting

| Message | Cause | Fix |
|---|---|---|
| `unknown shorthand flag: 't' in -t` | Docker's buildx plugin is missing | `sudo apt-get install docker-buildx-plugin` (the runner now says so) |
| `~/.docker/config.json: permission denied` | Docker was once run with sudo | harmless with sudo; or `sudo chown -R $USER:$(id -gn) ~/.docker` |
| Safari: "the address is invalid" on `bolt://neo4j:7687` | a setting in the log, shown as a link | nothing to do (only step 9 uses Neo4j) |
| Step 16 printed its command but no tables yet | it was still running when looked at | `bash standalone/run_hecores.sh --quick --step summary` |
| tmux: no new window with Control-b `c` | Control still held when pressing c | release both keys after b; or use `tmux new-window` from another tab (2b) |
| `tmux new -s hecores`: duplicate session | already created | `tmux attach -t hecores` |
| DrugBank login? | not needed | only step 2 (rebuilding from scratch) needs the DrugBank XML; never put the password in a file |

## 6. Times measured (November 2026, 8 cores)

| Step | Time |
|---|---|
| `--check` (first build) | a few minutes |
| `--quick` | ≈ 55 min (evaluation ≈ 16 min per experiment for the four test sets together) |
| `hcs`, full | ≈ 18 min permutations + ≈ 14 min null distributions + ≈ 4 min path scores |
| `splits` + `evaluate`, 100 experiments | estimate: ≈ 8–10 h with the three windows of 2b; over a day in one window |
