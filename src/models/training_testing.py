#coding: utf-8

import pandas as pd
import random
import numpy as np
import src.preprocessing.datasets as datasets
from collections import defaultdict
from scipy.sparse import coo_matrix
import json
from utils.utils import open_json
import os
import json
from utils.config import (PCT, SFX, HETMAT_DIR, N_EXPERIMENTS, N_EXPERIMENTS_MPS,
                          NEGATIVES, RQ3, CT_STATUS, CT_SFX, CT_STOPPED, describe)

# === 1. Precompute and Save ===

def precompute_and_save_pairs(main_dataset, other_datasets, n_experiments, test_size,
                              per_positives, n_negatives, n_negatives_per_positive, output_path,
                              negative_filter=None):
    """
    other_datasets: datasets for which training sets are written (as before).
    negative_filter: datasets in which a test negative must NOT be positive.
        None = other_datasets (published behaviour: negatives may not be
        positive in HCS_1 or any random control, i.e. pairs flagged by HCS
        cannot be negatives). See main() for what the revision uses.
    """
    if negative_filter is None:
        negative_filter = other_datasets

    def to_serializable_tuple(triple):
        # Ensure (user, item, label) are all native Python ints
        return (int(triple[0]), int(triple[1]), int(triple[2]))

    saved_sets = {}

    for i in range(n_experiments):
        random.seed(i)
        np.random.seed(i)
        print(i+1)

        # --- Compute test set ---
        test_set = get_test_set(
            main_dataset,
            test_size=test_size,
            per_positives=per_positives,
            n_negatives=n_negatives,
            seed=i,
            other_datasets=negative_filter
        )
        test_set_serializable = [to_serializable_tuple(t) for t in test_set]

        test_neg = [(a, b) for (a, b, c) in test_set if c == 0]
        test_pos = [(a, b) for (a, b, c) in test_set if c == 1]

        # --- Compute train sets ---
        train_sets = {}
        for dataset in [main_dataset] + other_datasets:
            train_pairs = get_train_set(
                dataset,
                test_pos,
                test_neg,
                n_negatives_per_positive=n_negatives_per_positive,
                seed=i
            )
            train_sets[dataset.name] = [to_serializable_tuple(t) for t in train_pairs]

        saved_sets[i] = {
            "test_set": test_set_serializable,
            "train_sets": train_sets
        }

        # --- Save every 10 iterations OR at the very end ---
        if (i + 1) % 10 == 0 or (i + 1) == n_experiments:
            tmp_path = f"{output_path}.{os.getpid()}.tmp"
            with open(tmp_path, "w") as f:
                json.dump(saved_sets, f)
                f.flush()              # flush Python buffer
                os.fsync(f.fileno())   # force OS write to disk
            os.replace(tmp_path, output_path)   # never leave a half-written file
            print(f"Progress saved to {output_path} (iteration {i+1})")

    print(f"All precomputed train/test pairs finished. Final saved to {output_path}")


# === 2. Load Saved Data ===

def load_precomputed_pairs(path):
    """
    Load precomputed train/test pairs from JSON.
    """
    with open(path, "r") as f:
        saved_sets = json.load(f)

    # Convert keys back to int (experiment index)
    saved_sets = {int(k): v for k, v in saved_sets.items()}
    return saved_sets

# === 3. Reconstruct DatasetSubsets from Pairs ===

def reconstruct_datasets_from_pairs(main, saved_sets, dataset_lookup, shape, create_coo_matrix):
    """
    Reconstruct DatasetSubset objects from saved (user, item, label) tuples.

    Parameters:
        saved_sets: dict loaded from JSON
        dataset_lookup: dict mapping dataset names to full dataset objects
        shape: shape of the full ratings matrix (e.g., main_dataset.ratings.shape)
        create_coo_matrix: function to convert (a, b) pairs into sparse COO matrix
    """
    reconstructed = {}

    for i, sets in saved_sets.items():
        # --- Test set ---
        test_set = sets["test_set"]
        test_pairs = [(a, b) for (a, b, _) in test_set]
        coo_test = create_coo_matrix(test_pairs, shape)
        test_dataset = dataset_lookup[main].subset(coo_test, subset_name="Test")

        # --- Train sets ---
        train_datasets = {}
        for name, train_pairs in sets["train_sets"].items():
            if name in ('CT12', 'CT34'):
                continue
            train_pairs_simple = [(a, b) for (a, b, _) in train_pairs]
            coo_train = create_coo_matrix(train_pairs_simple, shape)
            train_dataset = dataset_lookup[name].subset(coo_train, subset_name="TRAIN_" + name)
            train_datasets[name] = train_dataset

        reconstructed[i] = {
            "test_dataset": test_dataset,
            "train_datasets": train_datasets
        }

    return reconstructed

def get_test_set(dataset, test_size=0.2, per_positives=0.2, n_negatives=100, seed=0, other_datasets=[]):
    """
    Generates a test set of (drug_index, disease_index) positive-rating pairs,
    while ensuring every drug and every disease retains at least one positive rating
    for training. Also ensures selected negative pairs are 0 in all datasets.

    Parameters
    ----------
    dataset : Dataset
        The main Dataset instance.
    test_size : float
        Fraction of all positive ratings to include in the test set.
    per_positives : float
        Max fraction of a disease's positives to use for testing.
    n_negatives : int
        Number of negative samples to generate per disease in test set.
    seed : int
        Random seed for reproducibility.
    other_datasets : list of Dataset
        Additional Dataset instances where negative pairs must also be 0.

    Returns
    -------
    List[Tuple[int, int, int]]
        Selected test set as list of (drug_index, disease_index, label) tuples.
    """
    random.seed(seed)

    # Count positive ratings per drug and disease
    drug_counts = defaultdict(int)
    disease_counts = defaultdict(int)
    for drug_idx, disease_idx in dataset.get_positive_indices():
        drug_counts[drug_idx] += 1
        disease_counts[disease_idx] += 1

    total_positives = sum(drug_counts.values())
    n_test = int(total_positives * test_size)

    # Shuffle diseases
    disease_indices = list(range(dataset.nusers))
    random.shuffle(disease_indices)

    test_pairs = []
    disease_iter = 0

    while len(test_pairs) < n_test:
        if disease_iter >= len(disease_indices):
            print("Warning: Ran out of diseases before reaching desired test set size.")
            break

        d_idx = disease_indices[disease_iter]
        positive_drugs = dataset.get_positive_rows_for_disease(d_idx)

        if not positive_drugs:
            disease_iter += 1
            continue

        max_pairs = max(1, int(round(len(positive_drugs) * per_positives)))
        random.shuffle(positive_drugs)

        selected = 0
        for drug_idx in positive_drugs:
            if drug_counts[drug_idx] > 1 and disease_counts[d_idx] > 1:
                test_pairs.append((drug_idx, d_idx))
                drug_counts[drug_idx] -= 1
                disease_counts[d_idx] -= 1
                selected += 1

                if len(test_pairs) >= n_test or selected >= max_pairs:
                    break

        disease_iter += 1

    # Safeguard
    for drug_idx, cnt in drug_counts.items():
        assert cnt >= 1, f"Drug {drug_idx} lost all positive ratings!"
    for d_idx, cnt in disease_counts.items():
        assert cnt >= 1, f"Disease {d_idx} lost all positive ratings!"

    # Generate negatives — must be 0 in ALL datasets
    all_positives_set = set(dataset.get_positive_indices())
    test_pairs_set = set(test_pairs)
    test_disease_indices = set(d_idx for _, d_idx in test_pairs)
    negative_pairs = []

    for d_idx in test_disease_indices:
        negatives = set()
        attempts = 0

        while len(negatives) < n_negatives and attempts < dataset.nitems * 10:
            drug_idx = random.randint(0, dataset.nitems - 1)
            pair = (drug_idx, d_idx)

            if pair in all_positives_set or pair in test_pairs_set or pair in negatives:
                attempts += 1
                continue

            # Check all datasets (must NOT be positive in any)
            is_positive_in_others = any(
                other.is_positive_pair(drug_idx, d_idx, ignore_nan=True) == 1
                for other in other_datasets
            )
            if is_positive_in_others:
                attempts += 1
                continue

            negatives.add(pair)
            attempts += 1

        if len(negatives) < n_negatives:
            print(f"Warning: Only found {len(negatives)} negatives for disease {d_idx}")

        negative_pairs.extend(negatives)

    # Final labeled test set
    labeled_pairs = [(d, u, 1) for d, u in test_pairs] + [(d, u, 0) for d, u in negative_pairs]
    return labeled_pairs

def get_train_set(dataset, test_pairs, test_negatives, n_negatives_per_positive=5, seed=0):
    """
    Generates a training set of positive and sampled negative (drug, disease) pairs.

    Parameters
    ----------
    dataset : Dataset
        The stanscofi.Dataset instance.
    test_pairs : List[Tuple[int, int]]
        Positive test pairs returned by get_test_set.
    test_negatives : List[Tuple[int, int]]
        Negative test pairs returned by get_test_set.
    n_negatives_per_positive : int
        Number of negative samples to add per positive training pair for each disease.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    List[Tuple[int, int, int]]
        A list of (drug_idx, disease_idx, label) pairs for training.
    """

    random.seed(seed)

    test_pos_set = set(test_pairs)
    test_neg_set = set(test_negatives)
    all_pos_set = set(dataset.get_positive_indices())

    # Get all remaining positive pairs for training
    train_pos_set = all_pos_set - test_pos_set
    train_pairs = [(d, u, 1) for d, u in train_pos_set]

    # Group positives by disease (user)
    pos_by_disease = defaultdict(list)
    for drug_idx, disease_idx in train_pos_set:
        pos_by_disease[disease_idx].append(drug_idx)

    # Sample negatives per disease
    neg_samples = []
    for disease_idx, positive_drugs in pos_by_disease.items():
        possible_negatives = set(range(dataset.nitems))
        used_pairs = set((d, disease_idx) for d in positive_drugs) | test_pos_set | test_neg_set
        available_negatives = list(possible_negatives - set(d for d, u in used_pairs if u == disease_idx))

        for _ in range(len(positive_drugs) * n_negatives_per_positive):
            if not available_negatives:
                break
            drug_idx = random.choice(available_negatives)
            neg_samples.append((drug_idx, disease_idx, 0))
            available_negatives.remove(drug_idx)  # avoid duplicates

    train_pairs.extend(neg_samples)
    return train_pairs

def create_coo_matrix(indices, shape):
    """
    Create a COO sparse matrix from a list of indices and a given shape.

    Parameters:
    - indices: List of (row, col) tuples where the value should be 1
    - shape: Tuple (nrows, ncols) defining the shape of the matrix

    Returns:
    - A scipy.sparse.coo_matrix
    """
    if not indices:
        return coo_matrix(shape)

    rows, cols = zip(*indices)  # separate the tuple pairs
    data = [1] * len(indices)   # all values are 1

    return coo_matrix((data, (rows, cols)), shape=shape)

def build_dataset_from_pairs(drug_disease_pairs, users, items, name, same_item_user_features=False):
    """
    Creates a Dataset object with ratings = 1 for specified (disease, drug) pairs.

    Parameters
    ----------
    drug_disease_pairs : List[Tuple[str, str]]
        List of (disease_id, drug_id) pairs for which the rating should be 1.
    users : pd.DataFrame or np.ndarray
        User feature matrix.
    items : pd.DataFrame or np.ndarray
        Item feature matrix.
    same_item_user_features : bool
        Whether user and item features are the same.
    name : str
        Name of the dataset.

    Returns
    -------
    Dataset
        A Dataset instance with ratings as specified.
    """
    # Infer lists of IDs
    if isinstance(users, pd.DataFrame):
        user_ids = list(users.columns)
    else:
        user_ids = [str(i) for i in range(users.shape[1])]

    if isinstance(items, pd.DataFrame):
        item_ids = list(items.columns)
    else:
        item_ids = [str(i) for i in range(items.shape[1])]

    # Build empty rating matrix
    ratings = np.zeros((len(item_ids), len(user_ids)))

    # Fill in 1s for specified (disease, drug) pairs
    for disease_id, drug_id in drug_disease_pairs:
        if disease_id not in user_ids:
            raise ValueError(f"Disease ID {disease_id} not found in user_ids.")
        if drug_id not in item_ids:
            raise ValueError(f"Drug ID {drug_id} not found in item_ids.")
        drug_idx = item_ids.index(drug_id)
        disease_idx = user_ids.index(disease_id)
        ratings[drug_idx, disease_idx] = 1

    # Create DataFrame version for compatibility
    ratings_df = pd.DataFrame(ratings, index=item_ids, columns=user_ids)

    # Construct the Dataset
    dataset = datasets.Dataset(ratings=ratings_df, users=users, items=items, 
                      same_item_user_features=same_item_user_features, name=name)
    
    return dataset

def unique_metapath_pairs(file_path, metapath, base_pairs=None, known_pairs=frozenset()):
    """
    Pairs to keep when metapath `metapath` is removed (RQ3), as (condition, drug).

    file_path: all_dwpc_results_with_p.tsv (one row per pair and metapath).
    A pair is unique to `metapath` if that is its only significant metapath
    (adjusted p <= 0.05).

    Published version (base_pairs=None): all significant pairs minus those
    unique to `metapath`. This set lacks the known drug-condition pairs that
    are not significant themselves (1,902 of the 4,478 on the full graph),
    while the HCS_1 it is compared with contains them.
    Revision (base_pairs = HCS_1 pairs): HCS_1 minus the pairs unique to
    `metapath`, keeping known pairs (known_pairs), i.e. HCS_1 as if that
    metapath did not exist -- the analysis described in the thesis.
    """
    df = pd.read_csv(file_path, sep="\t", usecols=["source_id", "target_id", "metapath", "adjusted_p_value"],
                     dtype=str)
    df = df[df["adjusted_p_value"].astype(float) <= 0.05]
    all_pairs = set(zip(df["target_id"], df["source_id"]))
    n_mps = df.groupby(["target_id", "source_id"])["metapath"].nunique()
    only_one = set(n_mps[n_mps == 1].index)
    of_this = set(zip(df.loc[df["metapath"] == metapath, "target_id"], df.loc[df["metapath"] == metapath, "source_id"]))
    unique_to_this = only_one & of_this
    if base_pairs is None:
        return all_pairs - unique_to_this
    return set(base_pairs) - (unique_to_this - set(known_pairs))


def ct_pair_sets(ct_file, drugs_file, exclude_status=None):
    """Clinical-trial test pairs (condition, drug), phases I/II and III/IV,
    minus known indications -- as in the published code.

    exclude_status: set of trial statuses (e.g. {"Withdrawn", "Terminated", "Suspended"}).
        If given, a pair is kept only if at least one trial of the same phase set that
        supports it has another status (sensitivity analysis, HECORES_CT_STATUS=active)."""
    ct = open_json(ct_file)
    ct_pairs12, ct_pairs34 = set(), set()
    for k, v in ct.items():
        if exclude_status and (v.get('overall_status') or '') in exclude_status:
            continue
        conditions = v['conditions']
        drugs = v['drugs']
        phase = v['phase']
        if phase:
            if '1' in phase or '2' in phase:
                for c in conditions:
                    for d in drugs:
                        ct_pairs12.add((c, d))
            elif '3' in phase or '4' in phase:
                for c in conditions:
                    for d in drugs:
                        ct_pairs34.add((c, d))
    drugs = open_json(drugs_file)
    orig_pairs = {(c, k) for k, v in drugs.items() for c in v['treats']}
    return ct_pairs12 - orig_pairs, ct_pairs34 - orig_pairs


n_metapaths = {
    'Dt>PRa>CO': 1,
    'Dh>CA<hDi>CO': 6,
    'Dh>CH<hDi>CO': 6,
    'Di>CO<aPRa>CO': 6,
    'Dh>PH<hDi>CO': 6,
    'Dh>T<hDi>CO': 6,
    'Dt>PR<tDi>CO': 6,
    'Dt>PRh>B<hPRa>CO': 3,
    'Dt>PRh>CE<hPRa>CO': 3,
    'Dt>PRh>M<hPRa>CO': 3
}


def load_datasets():
    """HCS_0, HCS_1, the three random controls and the similarity matrices
    of the current graph (HECORES_PCT)."""
    users = pd.read_csv(f'cond_sims{SFX}.csv', index_col=0)
    items = pd.read_csv(f'drug_sims{SFX}.csv', index_col=0)
    out = {}
    for name, f in [("HCS_0", "hcs0"), ("HCS_1", "hcs1"), ("HCS_RANDOM1", "hcs_random1"),
                    ("HCS_RANDOM2", "hcs_random2"), ("HCS_RANDOM3", "hcs_random3")]:
        ratings = pd.read_csv(f'ratings_{f}{SFX}.csv', index_col=0)
        out[name] = datasets.Dataset(ratings=ratings, users=users, items=items,
                                     same_item_user_features=False, name=name)
    return out, users, items


def main(sets=("main", "ct", "mps")):
    """
    Precomputes the train/test pairs of every experiment.

    main  test = 20% of HCS_0, train on each dataset   -> train_test_pairs{sfx}{n}.json
    ct    test = clinical-trial pairs I/II and III/IV  -> train_test_pairs_ct12_{n}.json, ..._ct34_{n}.json
    mps   RQ3, HCS_1 without each metapath             -> train_test_pairs_mps_{n}.json
    (ct and mps only on the full graph, as in the paper)

    Test negatives (HECORES_NEGATIVES):
      independent  (revision) a negative may not be a known positive of the test set's own
                   source: HCS_0 for the DrugBank and RQ3 test sets, HCS_0 + the clinical-trial
                   pairs for the clinical-trial test sets
      paper        a negative may also not be positive in HCS_1 or any random control (or any
                   RQ3 matrix), so pairs that HCS flagged can never be test negatives
    """
    print(describe())
    d, users, items = load_datasets()
    hcs_0, hcs_1 = d["HCS_0"], d["HCS_1"]
    others = [d["HCS_1"], d["HCS_RANDOM1"], d["HCS_RANDOM2"], d["HCS_RANDOM3"]]
    paper = NEGATIVES == "paper"
    n = N_EXPERIMENTS

    if "main" in sets:
        precompute_and_save_pairs(
            main_dataset=hcs_0,
            other_datasets=others,
            n_experiments=n,
            test_size=0.2,
            per_positives=0.2,
            n_negatives=100,
            n_negatives_per_positive=5,
            output_path=f"train_test_pairs{SFX}{n}.json",
            negative_filter=None if paper else []
        )

    if PCT:
        return

    if "ct" in sets:
        if not os.path.exists('clinical_trials.json'):
            print("clinical_trials.json not found: clinical-trial test sets skipped")
        else:
            all12, all34 = ct_pair_sets('clinical_trials.json', "drugs_with_conditions_final.json")
            print(f"clinical-trial pairs: {len(all12)} (phases I/II), {len(all34)} (phases III/IV)")
            active = CT_STATUS == "active"
            if active:
                ct_pairs12, ct_pairs34 = ct_pair_sets('clinical_trials.json', "drugs_with_conditions_final.json",
                                                      exclude_status=CT_STOPPED)
                print(f"  without pairs supported only by withdrawn/terminated/suspended trials: "
                      f"{len(ct_pairs12)} (phases I/II), {len(ct_pairs34)} (phases III/IV)")
            else:
                ct_pairs12, ct_pairs34 = all12, all34
            for name, pairs, all_pairs in (("CT12", ct_pairs12, all12), ("CT34", ct_pairs34, all34)):
                ct_ds = build_dataset_from_pairs(list(pairs), users, items, name, same_item_user_features=False)
                if active:
                    # the excluded trial pairs must not become test negatives
                    all_ds = build_dataset_from_pairs(list(all_pairs), users, items, f"{name}_ALL",
                                                      same_item_user_features=False)
                    neg_filter = ([hcs_0] + others if paper else [hcs_0]) + [all_ds]
                else:
                    neg_filter = None if paper else [hcs_0]
                precompute_and_save_pairs(
                    main_dataset=ct_ds,
                    other_datasets=[hcs_0] + others,
                    n_experiments=n,
                    test_size=0.2,
                    per_positives=0.2,
                    n_negatives=100,
                    n_negatives_per_positive=5,
                    output_path=f"train_test_pairs_{name.lower()}{CT_SFX}_{n}.json",
                    negative_filter=neg_filter
                )

    if "mps" in sets:
        f = f'{HETMAT_DIR}/path-counts/dwpc_results/all_dwpc_results_with_p.tsv'
        if not os.path.exists(f):
            print(f"{f} not found: RQ3 (metapaths) skipped")
            return
        hcs1_pairs = {(hcs_1.user_list[u], hcs_1.item_list[i]) for i, u in hcs_1.get_positive_indices()}
        known = {(hcs_0.user_list[u], hcs_0.item_list[i]) for i, u in hcs_0.get_positive_indices()}
        mps = []
        for mp in n_metapaths:
            pairs = unique_metapath_pairs(f, mp, base_pairs=None if RQ3 == "paper" else hcs1_pairs,
                                          known_pairs=known)
            print(f"  without {mp}: {len(pairs)} positive pairs")
            mps.append(build_dataset_from_pairs(list(pairs), users, items, mp, same_item_user_features=False))
        precompute_and_save_pairs(
            main_dataset=hcs_0,
            other_datasets=[hcs_1] + mps,
            n_experiments=N_EXPERIMENTS_MPS,
            test_size=0.2,
            per_positives=0.2,
            n_negatives=100,
            n_negatives_per_positive=5,
            output_path=f"train_test_pairs_mps_{N_EXPERIMENTS_MPS}.json",
            negative_filter=None if paper else []
        )


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Precompute train/test pairs (see main()).")
    ap.add_argument("--sets", nargs="+", default=["main", "ct", "mps"], choices=["main", "ct", "mps"])
    main(ap.parse_args().sets)
