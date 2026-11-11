# Expected output of the installation test

These four tables come from `bash standalone/run_hecores.sh --quick`, run on the lab server on 9 Nov 2026 (code of branch `revision`, Docker image `hecores-dr:core`). On a second machine the same command gave the same RQ3 table (`summary_mps.csv`).

The quick test runs the whole pipeline with 2 permutations and 2 experiments, so **its numbers have no scientific meaning**. They only show that an installation computes the same thing.

After a quick test, compare:

```bash
diff data/quick/eval_revision/summary_main.csv standalone/expected/quick/summary_main.csv
diff data/quick/eval_revision/summary_ct12.csv standalone/expected/quick/summary_ct12.csv
diff data/quick/eval_revision/summary_ct34.csv standalone/expected/quick/summary_ct34.csv
diff data/quick/eval_revision/summary_mps.csv  standalone/expected/quick/summary_mps.csv
```

No output means identical. All random steps use fixed seeds, so with the same package versions (`requirements.txt`) the tables should match exactly. On different hardware or package versions, differences in the last digit can occur. Larger differences mean something in the installation differs; look at the package versions printed by `run_hecores.sh --check`.

The test starts from the student's intermediate files in `data/work/` (see `standalone/README.md`, "Starting from intermediate files").
