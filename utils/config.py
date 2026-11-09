"""
Settings shared by all pipeline scripts, read from environment variables.

The scripts read and write their intermediate files (conditions.json,
drugs.json, graph.json, ratings_*.csv, ...) in the CURRENT folder, as before.
Run them from the work folder, e.g. with standalone/run_hecores.sh, which
sets everything below.

Environment variables (all optional)
  HECORES_RAW            folder with the downloaded source files (default: current folder)
                         CTD_diseases.csv, CTD_genes.csv, CTD_curated_genes_diseases.csv,
                         "WHO ATC-DDD 2024-07-31.csv", "full database.xml" (DrugBank),
                         go.obo, desc2025.xml (MeSH; downloaded if absent),
                         ctg-public-xml.zip (ClinicalTrials.gov)
  HECORES_PCT            ablation (RQ2): percentage of edges removed, e.g. 20; empty = full graph
  HECORES_N_EXPERIMENTS  experiments per model (default 100; BNNR 5, metapaths 50)
  HECORES_MODELS         comma-separated models to evaluate (default ALSWR,PMF,LogisticMF,HCS;
                         HCS = HCS alone, no collaborative filtering; add BNNR, which needs Octave)
  HECORES_N_PERMS        HCS permutations (default 200)
  HECORES_JOBS           CPU cores for HCS (default 2)
  NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD   Neo4j connection (default bolt://localhost:7687,
                         neo4j, password)

Corrections made for the revision (see CHANGES_REVISION.md). Each can be
switched back to the published behaviour, to reproduce the paper's numbers:
  HECORES_METRICS     scores (default) | labels   -- rank metrics on model scores,
                                                     or on thresholded +1/-1 labels (published)
  HECORES_NEGATIVES   independent (default) | paper -- test negatives exclude known positives
                                                     only, or also every HCS/random positive (published)
  HECORES_RQ3         corrected (default) | paper -- RQ3 matrices = HCS_1 minus the unique pairs
                                                     of one metapath, or significant pairs only (published)
  HECORES_PAPER_MODE=1  sets all three to the published behaviour
"""
import os

RAW_DIR = os.environ.get("HECORES_RAW", ".")

PCT = os.environ.get("HECORES_PCT", "").strip()
SFX = f"_{PCT}pct" if PCT else ""            # suffix of the student's file names
HETMAT_DIR = f"graph.hetmat{PCT}"            # graph.hetmat (full) or graph.hetmat20 (20% removed)

N_EXPERIMENTS = int(os.environ.get("HECORES_N_EXPERIMENTS", "100"))
N_EXPERIMENTS_BNNR = min(5, N_EXPERIMENTS)
N_EXPERIMENTS_MPS = min(50, N_EXPERIMENTS)
MODELS = [m.strip() for m in os.environ.get("HECORES_MODELS", "ALSWR,PMF,LogisticMF,HCS").split(",") if m.strip()]
N_PERMS = int(os.environ.get("HECORES_N_PERMS", "200"))
JOBS = int(os.environ.get("HECORES_JOBS", "2"))

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")

PAPER_MODE = os.environ.get("HECORES_PAPER_MODE", "0") == "1"
METRICS = "labels" if PAPER_MODE else os.environ.get("HECORES_METRICS", "scores")
NEGATIVES = "paper" if PAPER_MODE else os.environ.get("HECORES_NEGATIVES", "independent")
RQ3 = "paper" if PAPER_MODE else os.environ.get("HECORES_RQ3", "corrected")
assert METRICS in ("scores", "labels"), METRICS
assert NEGATIVES in ("independent", "paper"), NEGATIVES
assert RQ3 in ("corrected", "paper"), RQ3


def raw(name):
    """Path of a downloaded source file."""
    return os.path.join(RAW_DIR, name)


def describe():
    return (f"pct={PCT or 'full graph'} metrics={METRICS} negatives={NEGATIVES} rq3={RQ3} "
            f"n_experiments={N_EXPERIMENTS} models={','.join(MODELS)}")
