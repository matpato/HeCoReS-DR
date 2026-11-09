#!/bin/bash
set -e

# =============================================================================
# HeCoReS-DR -- pipeline runner (same layout as MedJsonify's standalone runner)
# =============================================================================
#
# Runs every step of the pipeline, each in its own Docker container (or, with
# --local, with the python3 of your own environment). A step whose input files
# are missing is skipped with a message, so the pipeline can start from the
# source files (step 1) or from intermediate files (e.g. step 8 or 11).
# A log of every run is written to standalone/logs/.
#
# USAGE (from the HeCoReS-DR main folder)
#   bash standalone/run_hecores.sh                       -- all steps, in order
#   bash standalone/run_hecores.sh --list                -- list the steps and their input files
#   bash standalone/run_hecores.sh --status              -- which input files are present
#   bash standalone/run_hecores.sh --step hcs            -- one step (name or number)
#   bash standalone/run_hecores.sh --from matrices       -- from a step to the end
#   bash standalone/run_hecores.sh --paper               -- steps 14-16 as published (metrics on
#                                                           labels, filtered negatives, RQ3 as
#                                                           published), to reproduce the paper
#   bash standalone/run_hecores.sh --pct "20 40 60 80"   -- ablation (RQ2): steps 12-16 for each
#                                                           percentage of removed edges
#   bash standalone/run_hecores.sh --quick               -- installation test (minutes): 2
#                                                           experiments, 2 permutations, in data/quick
#   bash standalone/run_hecores.sh --n-experiments 50    -- fewer experiments (default 100)
#   bash standalone/run_hecores.sh --models "ALSWR HCS"  -- models for step 15 (default
#                                                           ALSWR PMF LogisticMF HCS; BNNR needs Octave)
#   bash standalone/run_hecores.sh --sets "main mps"     -- test sets for step 15 (default
#                                                           main ct12 ct34 mps)
#   bash standalone/run_hecores.sh --jobs 4              -- CPU cores for step 12 (default 2)
#   bash standalone/run_hecores.sh --local               -- no Docker (pip install -r requirements.txt)
#   bash standalone/run_hecores.sh --sudo                -- use "sudo docker" (servers where docker needs sudo);
#                                                           same as DOCKER="sudo docker" bash standalone/run_hecores.sh
#   bash standalone/run_hecores.sh --check               -- check that the packages import
#   bash standalone/run_hecores.sh --shell 'COMMAND'     -- run a command in the container (in data/work)
#   bash standalone/run_hecores.sh --rebuild --check     -- rebuild the image first
#   bash standalone/run_hecores.sh --dry-run             -- print the commands only
#
# STEPS (in order)            script                                  needs (raw = data/raw)
#   1  ctd         src/preprocessing/ctd_processing.py         raw CTD_diseases.csv
#   2  drugbank    src/preprocessing/drugbank_processing.py    raw "full database.xml" (DrugBank 5.1.13)
#   3  mesh        src/preprocessing/mesh.py                   drugs.json (downloads MeSH 2025, or raw desc2025.xml)
#   4  conditions  src/preprocessing/extract_conditions.py     drugs.json conditions.json (HunFlair2; image hecores-dr:ner)
#   5  atc         src/preprocessing/atc_codes_processing.py   raw "WHO ATC-DDD 2024-07-31.csv", drugs_with_conditions.json
#   6  proteins    src/kg/condition_associations.py            raw CTD_genes.csv CTD_curated_genes_diseases.csv, proteins.json
#   7  go          src/preprocessing/go_target_processing.py   raw go.obo, proteins.json targets.json
#   8  filter      src/kg/processing_for_neo4j.py              drugs_with_conditions.json conditions.json
#   9  neo4j       src/kg/neo4j.py                             outputs of steps 2-8 (starts a Neo4j container)
#   10 trials      src/preprocessing/clinical_trials.py        raw ctg-public-xml.zip, outputs of step 8
#   11 graph       src/models/kg_enhanced/hpy.py               graph_nodes.json graph_triples.txt
#   12 hcs         src/models/kg_enhanced/hmp.py               graph.json (about 45 min on 2 cores)
#   13 matrices    src/models/similarity_based/sims.py         outputs of steps 8 and 12
#   14 splits      src/models/training_testing.py              outputs of step 13 (+ clinical_trials.json)
#   15 evaluate    src/evaluation/scofi.py                     outputs of step 14
#   16 summary     src/evaluation/summarize.py                 outputs of step 15
#
# FOLDERS
#   data/raw/                      downloaded source files (HECORES_RAW)
#   data/work/                     intermediate files of steps 1-13 (scripts run here)
#   data/work/eval_revision/       steps 14-16, corrections of the revision (default)
#   data/work/eval_paper/          steps 14-16 with --paper
#
# Monitoring a run:
#   tail -f standalone/logs/$(ls -t standalone/logs/ | head -1)
# =============================================================================


# -----------------------------------------------------------------------
# 0. Paths and settings (override any of them with environment variables)
# -----------------------------------------------------------------------
STANDALONE=$(cd "$(dirname "$0")" && pwd -P)
ROOT=${ROOT:-$(dirname "$STANDALONE")}
cd "$ROOT"

IMAGE=${IMAGE:-hecores-dr}
DOCKERFILE=${DOCKERFILE:-$STANDALONE/Dockerfile}
for arg in "$@"; do [ "$arg" == "--sudo" ] && DOCKER="sudo docker"; done
if [ -z "$DOCKER" ]; then
    # plain docker only if it can reach the Docker daemon; otherwise sudo docker
    if docker version --format '{{.Server.Version}}' &>/dev/null; then DOCKER=docker; else DOCKER="sudo docker"; fi
fi
DOCKER_USER=${DOCKER_USER-"$(id -u):$(id -g)"}

NEO4J_CONTAINER=${NEO4J_CONTAINER:-hecores_neo4j}
NEO4J_IMAGE=${NEO4J_IMAGE:-neo4j:5}
NEO4J_AUTH=${NEO4J_AUTH:-neo4j/hecores_password}
NEO4J_NETWORK=${NEO4J_NETWORK:-hecores-net}
NEO4J_VOLUME=${NEO4J_VOLUME:-hecores_neo4j_data}

ALL_STEPS=(ctd drugbank mesh conditions atc proteins go filter neo4j trials graph hcs matrices splits evaluate summary)

step_name() {
    case "$1" in
        1) echo ctd ;;      2) echo drugbank ;;  3) echo mesh ;;     4) echo conditions ;;
        5) echo atc ;;      6) echo proteins ;;  7) echo go ;;       8) echo filter ;;
        9) echo neo4j ;;    10) echo trials ;;   11) echo graph ;;   12) echo hcs ;;
        13) echo matrices ;; 14) echo splits ;;  15) echo evaluate ;; 16) echo summary ;;
        *) echo "$1" ;;
    esac
}

# -----------------------------------------------------------------------
# Arguments
# -----------------------------------------------------------------------
STEP="all"; FROM_STEP=""; DRY_RUN=0; LIST=0; STATUS=0; CHECK=0; SHELL_CMD=""; REBUILD=0
LOCAL=0; QUICK=0; PAPER=0; PCTS=""; N_EXP=100; N_PERMS=200; JOBS=2
MODELS="ALSWR PMF LogisticMF HCS"; SETS="main ct12 ct34 mps"
WORK=${WORK:-}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --step)          STEP=$(step_name "$2"); shift 2 ;;
        --from)          FROM_STEP=$(step_name "$2"); shift 2 ;;
        --paper)         PAPER=1; shift ;;
        --pct)           PCTS="$2"; shift 2 ;;
        --quick)         QUICK=1; N_EXP=2; N_PERMS=2; shift ;;
        --n-experiments) N_EXP="$2"; shift 2 ;;
        --models)        MODELS="$2"; shift 2 ;;
        --sets)          SETS="$2"; shift 2 ;;
        --jobs)          JOBS="$2"; shift 2 ;;
        --work)          WORK="$2"; shift 2 ;;
        --local)         LOCAL=1; shift ;;
        --sudo)          shift ;;   # handled above: DOCKER="sudo docker"
        --rebuild)       REBUILD=1; shift ;;
        --dry-run)       DRY_RUN=1; shift ;;
        --list)          LIST=1; shift ;;
        --status)        STATUS=1; shift ;;
        --check)         CHECK=1; shift ;;
        --shell)         SHELL_CMD="$2"; shift 2 ;;
        -h|--help)       sed -n '4,64p' "$0"; exit 0 ;;
        *)               echo "Unknown argument: $1"; exit 1 ;;
    esac
done

valid_step() { local s; for s in all "${ALL_STEPS[@]}"; do [ "$s" == "$1" ] && return 0; done; return 1; }
if ! valid_step "$STEP"; then echo "Unknown step: $STEP (use one of: ${ALL_STEPS[*]})"; exit 1; fi
if [ -n "$FROM_STEP" ] && ! valid_step "$FROM_STEP"; then echo "Unknown step: $FROM_STEP"; exit 1; fi

if [ $LIST == 1 ]; then
    sed -n '/^# STEPS/,/^# FOLDERS/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
    exit 0
fi

STEPS_TO_RUN=()
if [ -n "$FROM_STEP" ]; then
    started=0
    for s in "${ALL_STEPS[@]}"; do
        [ "$s" == "$FROM_STEP" ] && started=1
        [ $started == 1 ] && STEPS_TO_RUN+=("$s")
    done
elif [ "$STEP" == "all" ]; then
    STEPS_TO_RUN=("${ALL_STEPS[@]}")
else
    STEPS_TO_RUN=("$STEP")
fi

# -----------------------------------------------------------------------
# Folders (relative to the main folder)
# -----------------------------------------------------------------------
RAW_REL=data/raw
if [ -z "$WORK" ]; then
    if [ $QUICK == 1 ]; then WORK=data/quick; else WORK=data/work; fi
fi
WORK_REL=${WORK#"$ROOT/"}
MODE=$([ $PAPER == 1 ] && echo paper || echo revision)
EVAL_REL=$WORK_REL/eval_$MODE

if [ $DRY_RUN == 0 ]; then
    mkdir -p "$RAW_REL" "$WORK_REL" "$EVAL_REL"
    if [ $QUICK == 1 ] && [ "$WORK_REL" == "data/quick" ]; then
        # quick test: copy the starting files (not the results) from data/work
        for f in drugs_with_conditions.json conditions_all.json conditions.json graph_nodes.json \
                 graph_triples.txt clinical_trials.json; do
            [ -f "data/work/$f" ] && [ ! -f "data/quick/$f" ] && cp "data/work/$f" data/quick/
        done
    fi
fi

# -----------------------------------------------------------------------
# Logging (screen and file)
# -----------------------------------------------------------------------
LOG_FILE=""
if [ $DRY_RUN == 0 ] && [ $CHECK == 0 ] && [ $STATUS == 0 ] && [ -z "$SHELL_CMD" ]; then
    mkdir -p "$STANDALONE/logs"
    LOG_FILE=$STANDALONE/logs/run_$(date +%Y%m%d_%H%M%S).log
    echo "Logging to $LOG_FILE"
    exec > >(tee -a "$LOG_FILE") 2>&1
fi

run() { echo "+ $*"; if [ $DRY_RUN == 0 ]; then "$@"; fi; }

# -----------------------------------------------------------------------
# Python: in a container (default) or local (--local)
# -----------------------------------------------------------------------
ENV_ARGS=(HECORES_RAW HECORES_N_EXPERIMENTS HECORES_N_PERMS HECORES_JOBS HECORES_MODELS
          HECORES_PAPER_MODE HECORES_PCT NEO4J_URI NEO4J_USER NEO4J_PASSWORD)
export HECORES_N_EXPERIMENTS=$N_EXP HECORES_N_PERMS=$N_PERMS HECORES_JOBS=$JOBS
export HECORES_MODELS=$(echo $MODELS | tr ' ' ',')
export HECORES_PAPER_MODE=$PAPER
export NEO4J_USER=${NEO4J_AUTH%%/*} NEO4J_PASSWORD=${NEO4J_AUTH#*/}

if [ $LOCAL == 1 ]; then
    BASE=$ROOT
    export HECORES_RAW=$ROOT/$RAW_REL
    export NEO4J_URI=${NEO4J_URI:-bolt://localhost:7687}
    # py <dir> <variant> <python args...>
    py() { local dir=$1; shift 2; ( cd "$ROOT/$dir" && run env PYTHONPATH="$ROOT" python3 -W ignore "$@" ); }
    sh_in() { ( cd "$ROOT/$WORK_REL" && run bash -c "$1" ); }
    ENGINE="local python3 ($(python3 --version 2>&1))"
else
    BASE=/work
    export HECORES_RAW=/work/$RAW_REL
    export NEO4J_URI=${NEO4J_URI:-bolt://neo4j:7687}
    NETWORK=""
    py() {   # py <dir> <variant: core|ner|octave> <python args...>
        local dir=$1 variant=$2; shift 2
        local args=(--rm -v "$ROOT:/work" -w "/work/$dir" -e PYTHONPATH=/work -e PYTHONUNBUFFERED=1)
        # explicit values: "sudo docker" starts with an empty environment, so "-e NAME" alone would pass nothing
        for v in "${ENV_ARGS[@]}"; do args+=(-e "$v=${!v}"); done
        [ -n "$DOCKER_USER" ] && args+=(--user "$DOCKER_USER")
        [ -n "$NETWORK" ] && args+=(--network "$NETWORK")
        ensure_image "$variant"
        run $DOCKER run "${args[@]}" --entrypoint python3 "$IMAGE:$variant" -W ignore "$@"
    }
    sh_in() {
        local args=(--rm -v "$ROOT:/work" -w "/work/$WORK_REL")
        [ -n "$DOCKER_USER" ] && args+=(--user "$DOCKER_USER")
        ensure_image core
        run $DOCKER run "${args[@]}" --entrypoint bash "$IMAGE:core" -c "$1"
    }
    ENGINE="Docker ($DOCKER), image $IMAGE:core|ner|octave"
fi

BUILT=""
ensure_image() {   # build hecores-dr:<variant> if needed (once per run with --rebuild)
    local variant=$1 ner=0 octave=0
    [ "$variant" == "ner" ] && ner=1
    [ "$variant" == "octave" ] && octave=1
    [[ " $BUILT " == *" $variant "* ]] && return
    if [ $REBUILD == 1 ]; then $DOCKER rmi "$IMAGE:$variant" &>/dev/null || true; fi
    if [ $DRY_RUN == 1 ] || [ $REBUILD == 1 ] || ! $DOCKER image inspect "$IMAGE:$variant" &>/dev/null; then
        if [ $DRY_RUN == 0 ] && ! $DOCKER buildx version &>/dev/null; then
            echo "ERROR: 'docker build' is not available: recent Docker needs the buildx plugin."
            echo "  Check:    docker buildx version"
            echo "  Install:  sudo apt-get install docker-buildx-plugin   (Debian/Ubuntu, Docker's apt repository)"
            echo "            sudo dnf install docker-buildx-plugin       (Fedora/RHEL)"
            if [ -e "$HOME/.docker" ] && [ ! -r "$HOME/.docker/config.json" -o ! -w "$HOME/.docker" ]; then
                echo "  Also: $HOME/.docker is not yours (Docker was run with sudo), so Docker cannot read it:"
                echo "            sudo chown -R \"\$USER\":\"\$(id -gn)\" ~/.docker"
            fi
            echo "  Or run without Docker: pip install -r requirements.txt, then add --local"
            exit 1
        fi
        echo "Building $IMAGE:$variant"
        run $DOCKER build -t "$IMAGE:$variant" --build-arg WITH_NER=$ner --build-arg WITH_OCTAVE=$octave \
            -f "$DOCKERFILE" "$ROOT"
    fi
    BUILT="$BUILT $variant"
}

echo "========================================"
echo "HeCoReS-DR pipeline"
echo "Date:         $(date)"
echo "Steps:        ${STEPS_TO_RUN[*]}"
echo "Folder:       $ROOT"
echo "Work folder:  $WORK_REL   (evaluation: $EVAL_REL)"
echo "Python:       $ENGINE"
echo "Mode:         $MODE$([ $PAPER == 1 ] && echo ' (published metrics, negatives and RQ3 matrices)')"
echo "Experiments:  $N_EXP   Permutations: $N_PERMS   Jobs: $JOBS$([ $QUICK == 1 ] && echo '   (QUICK TEST)')"
echo "Models:       $MODELS   Test sets: $SETS"
[ -n "$PCTS" ] && echo "Ablation:     $PCTS % of edges removed (steps 12-16)"
echo "Commit:       $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "========================================"

if [ $CHECK == 1 ]; then
    py "$WORK_REL" core -c "import sys, numpy, pandas, scipy, sklearn, hetnetpy, hetmatpy, neo4j, lxml, pronto; \
print('Python', sys.version.split()[0]); \
[print(f'  {m.__name__:10s} {getattr(m, \"__version__\", \"\")}') for m in (numpy, pandas, scipy, sklearn, neo4j, lxml, pronto)]; \
import src.models.models, src.evaluation.validation, utils.config; print('HeCoReS-DR modules import OK')"
    exit $?
fi
if [ -n "$SHELL_CMD" ]; then sh_in "$SHELL_CMD"; exit $?; fi

# -----------------------------------------------------------------------
# Input files of each step (relative to the work folder; raw:NAME = data/raw/NAME)
# -----------------------------------------------------------------------
needs() {
    case "$1" in
        ctd)        echo "raw:CTD_diseases.csv" ;;
        drugbank)   echo "raw:full database.xml" ;;
        mesh)       echo "drugs.json|drug_categories.json" ;;
        conditions) echo "drugs.json|conditions.json" ;;
        atc)        echo "raw:WHO ATC-DDD 2024-07-31.csv|drugs_with_conditions.json" ;;
        proteins)   echo "raw:CTD_genes.csv|raw:CTD_curated_genes_diseases.csv|proteins.json" ;;
        go)         echo "raw:go.obo|proteins.json|targets.json" ;;
        filter)     echo "drugs_with_conditions.json|conditions.json" ;;
        neo4j)      echo "drugs_with_conditions_final.json|conditions.json|targets.json|proteins.json|go.json|atc.json|protein_map.json|go_name_to_id.json|drug_categories.json|mesh.json" ;;
        trials)     echo "raw:ctg-public-xml.zip|conditions.json|drugs_with_conditions_final.json" ;;
        graph)      echo "graph_nodes.json|graph_triples.txt" ;;
        hcs)        if [ -n "$PCT" ]; then echo "pruned_graphs/graph_stage_${PCT}pct.json|metagraph.json|nodes.tsv"; else echo "graph.json|metagraph.json|nodes.tsv"; fi ;;
        matrices)   echo "drugs_with_conditions_final.json|conditions.json|graph_triples${SFX}.txt|graph.hetmat${PCT}/path-counts/dwpc_results/all_dwpc_results_with_p.tsv" ;;
        splits)     echo "ratings_hcs0${SFX}.csv|ratings_hcs1${SFX}.csv|ratings_hcs_random1${SFX}.csv|cond_sims${SFX}.csv|drug_sims${SFX}.csv" ;;
        evaluate)   echo "ratings_hcs0${SFX}.csv|cond_sims${SFX}.csv|drug_sims${SFX}.csv" ;;
        summary)    echo "" ;;
    esac
}
missing_inputs() {   # prints the missing input files of a step
    local IFS='|' f
    for f in $(needs "$1"); do
        if [[ "$f" == raw:* ]]; then [ -e "$RAW_REL/${f#raw:}" ] || echo "data/raw/${f#raw:}"
        else [ -e "$WORK_REL/$f" ] || echo "$WORK_REL/$f"; fi
    done
}

if [ $STATUS == 1 ]; then
    for s in "${ALL_STEPS[@]}"; do
        PCT=""; SFX=""
        m=$(missing_inputs "$s" | tr '\n' ' ')
        if [ -z "$m" ]; then printf "  %-11s ready\n" "$s"; else printf "  %-11s missing: %s\n" "$s" "$m"; fi
    done
    exit 0
fi

# -----------------------------------------------------------------------
# Neo4j for step 9
# -----------------------------------------------------------------------
ensure_neo4j() {
    if [ $LOCAL == 1 ]; then echo "Using Neo4j at $NEO4J_URI (--local)"; return; fi
    if [ $DRY_RUN == 1 ]; then NETWORK=$NEO4J_NETWORK; echo "(dry run) would start $NEO4J_CONTAINER"; return; fi
    $DOCKER network inspect "$NEO4J_NETWORK" &>/dev/null || $DOCKER network create "$NEO4J_NETWORK" >/dev/null
    if $DOCKER container inspect "$NEO4J_CONTAINER" &>/dev/null; then
        [ "$($DOCKER inspect -f '{{.State.Running}}' "$NEO4J_CONTAINER")" == "true" ] || $DOCKER start "$NEO4J_CONTAINER" >/dev/null
    else
        echo "Starting Neo4j ($NEO4J_IMAGE) as $NEO4J_CONTAINER"
        $DOCKER run -d --name "$NEO4J_CONTAINER" --network "$NEO4J_NETWORK" --network-alias neo4j \
            -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH="$NEO4J_AUTH" -v "$NEO4J_VOLUME:/data" \
            "$NEO4J_IMAGE" >/dev/null
    fi
    NETWORK=$NEO4J_NETWORK
    echo -n "Waiting for Neo4j"
    for i in $(seq 1 60); do
        if $DOCKER exec "$NEO4J_CONTAINER" cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" "RETURN 1" &>/dev/null; then
            echo " -- ready"; return; fi
        echo -n "."; sleep 2
    done
    echo ""; echo "ERROR: Neo4j did not start in 2 minutes (docker logs $NEO4J_CONTAINER)"; exit 1
}

# -----------------------------------------------------------------------
# Steps
# -----------------------------------------------------------------------
header() { echo ""; echo "----------------------------------------"; echo "Step $1"; echo "----------------------------------------"; }

link_eval_inputs() {   # steps 14-16 run in eval_<mode>/ and read the matrices of the work folder
    [ $DRY_RUN == 1 ] && return
    ( cd "$EVAL_REL" && for f in ../ratings_*.csv ../cond_sims*.csv ../drug_sims*.csv ../clinical_trials.json \
                                  ../drugs_with_conditions_final.json ../graph.hetmat*; do
          [ -e "$f" ] && ln -sfn "$f" .
      done ) || true
}

run_step() {   # run_step <name>   (uses PCT/SFX for steps 12-16)
    local s=$1 num m
    for num in $(seq 1 ${#ALL_STEPS[@]}); do [ "${ALL_STEPS[$((num-1))]}" == "$s" ] && break; done
    header "$num $s${PCT:+ ($PCT% of edges removed)}"
    m=$(missing_inputs "$s")
    if [ -n "$m" ] && [ $DRY_RUN == 0 ]; then
        echo "Skipped -- missing input files:"; echo "$m" | sed 's/^/  /'; return
    fi
    case "$s" in
        ctd)        py "$WORK_REL" core -m src.preprocessing.ctd_processing ;;
        drugbank)   py "$WORK_REL" core -m src.preprocessing.drugbank_processing ;;
        mesh)       py "$WORK_REL" core -m src.preprocessing.mesh ;;
        conditions) py "$WORK_REL" ner  -m src.preprocessing.extract_conditions ;;
        atc)        py "$WORK_REL" core -m src.preprocessing.atc_codes_processing ;;
        proteins)   py "$WORK_REL" core -m src.kg.condition_associations ;;
        go)         py "$WORK_REL" core -m src.preprocessing.go_target_processing ;;
        filter)
            # processing_for_neo4j.py overwrites conditions.json with the filtered conditions;
            # keep the full version once and restore it, so the step can be run again
            if [ $DRY_RUN == 0 ]; then
                if [ -f "$WORK_REL/conditions_all.json" ]; then cp "$WORK_REL/conditions_all.json" "$WORK_REL/conditions.json"
                else cp "$WORK_REL/conditions.json" "$WORK_REL/conditions_all.json"; fi
            fi
            py "$WORK_REL" core -m src.kg.processing_for_neo4j ;;
        neo4j)      ensure_neo4j; py "$WORK_REL" core -m src.kg.neo4j ;;
        trials)     py "$WORK_REL" core -m src.preprocessing.clinical_trials ;;
        graph)      py "$WORK_REL" core -m src.models.kg_enhanced.hpy ;;
        hcs)        py "$WORK_REL" core -m src.models.kg_enhanced.hmp ;;
        matrices)   py "$WORK_REL" core -m src.models.similarity_based.sims ;;
        splits)
            link_eval_inputs
            local ts=(main)
            [ -z "$PCT" ] && [[ " $SETS " == *" ct1"* || " $SETS " == *" ct3"* ]] && ts+=(ct)
            [ -z "$PCT" ] && [[ " $SETS " == *" mps "* ]] && ts+=(mps)
            py "$EVAL_REL" core -m src.models.training_testing --sets "${ts[@]}" ;;
        evaluate)
            link_eval_inputs
            local sets=(main) variant=core
            [ -z "$PCT" ] && sets=($SETS)
            [[ " $MODELS " == *" BNNR "* ]] && variant=octave
            py "$EVAL_REL" $variant -m src.evaluation.scofi --sets "${sets[@]}" --models $MODELS ;;
        summary)    py "$EVAL_REL" core -m src.evaluation.summarize ;;
    esac
}

PCT=""; SFX=""; export HECORES_PCT=""
for s in "${STEPS_TO_RUN[@]}"; do
    case "$s" in
        hcs|matrices|splits|evaluate|summary)
            if [ -n "$PCTS" ]; then
                for PCT in $PCTS; do
                    SFX="_${PCT}pct"; export HECORES_PCT=$PCT
                    run_step "$s"
                done
                PCT=""; SFX=""; export HECORES_PCT=""
            else
                run_step "$s"
            fi ;;
        *) run_step "$s" ;;
    esac
done

echo ""
echo "Done: ${STEPS_TO_RUN[*]}"
echo "Results: $EVAL_REL/summary_*.csv"
[ -n "$LOG_FILE" ] && echo "Log: $LOG_FILE"
exit 0
