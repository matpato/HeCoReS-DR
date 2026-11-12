#!/bin/bash
# =============================================================================
# HeCoReS-DR -- run the whole pipeline on the sample (about 5 minutes)
# =============================================================================
#
# Copies sample/inputs/ to data/sample/ (ignored by git), runs steps 11-16
# (graph, hcs, matrices, splits, evaluate, summary) with 20 HCS permutations
# and 5 experiments, and compares the four summary tables with
# sample/expected/. See sample/README.md.
#
# USAGE (from the HeCoReS-DR main folder)
#   bash sample/run_sample.sh                 -- Docker (sudo docker, as the main runner)
#   bash sample/run_sample.sh --no-sudo       -- Docker without sudo (e.g. Docker Desktop on a Mac)
#   bash sample/run_sample.sh --local         -- your own Python (pip install -r requirements.txt)
#   bash sample/run_sample.sh --jobs 4        -- more CPU cores for step 12
#   bash sample/run_sample.sh --keep          -- keep data/sample/ from an earlier run
#
# Any other option is passed to standalone/run_hecores.sh. Changing
# --n-experiments or --n-perms gives different numbers from sample/expected/.
# =============================================================================
set -e

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

WORK=data/sample
N_EXP=5
N_PERMS=20
KEEP=0
PAPER=0
ARGS=()
for arg in "$@"; do
    if [ "$arg" == "--keep" ]; then KEEP=1; else ARGS+=("$arg"); fi
    [ "$arg" == "--paper" ] && PAPER=1
done

if [ $KEEP == 0 ]; then
    rm -rf "$WORK"
fi
mkdir -p "$WORK"
for f in sample/inputs/*; do
    [ -e "$WORK/$(basename "$f")" ] || cp "$f" "$WORK/"
done

bash standalone/run_hecores.sh --work "$WORK" --from graph \
    --n-experiments $N_EXP --n-perms $N_PERMS "${ARGS[@]}"

if [ $PAPER == 1 ]; then
    echo ""
    echo "--paper: tables in $WORK/eval_paper/; sample/expected/ holds the default (revision) run only."
    exit 0
fi

echo ""
echo "========================================"
echo "Comparison with sample/expected/"
echo "========================================"
status=0
for f in sample/expected/summary_*.csv; do
    name=$(basename "$f")
    out="$WORK/eval_revision/$name"
    if [ ! -f "$out" ]; then
        echo "  $name: MISSING (the run did not produce it)"; status=1
    elif diff -q "$f" "$out" >/dev/null; then
        echo "  $name: identical"
    else
        echo "  $name: DIFFERENT  (diff $f $out)"; status=1
    fi
done
if [ $status == 0 ]; then
    echo "All four tables are identical to the expected output."
else
    echo "Some tables differ; see sample/README.md, 'If the tables differ'."
fi
exit $status
