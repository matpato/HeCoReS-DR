import numpy as np
import pandas as pd
import csv
from utils.utils import open_json
import random
import src.preprocessing.datasets as datasets
import src.evaluation.validation as validation
import src.models.models as models
import random
import csv
from time import time

pct = 80
print(pct)

ratings_hcs0 = pd.read_csv(f'ratings_hcs0_{pct}pct.csv', index_col=0)
ratings_hcs1 = pd.read_csv(f'ratings_hcs1_{pct}pct.csv', index_col=0)
ratings_hcs_random1 = pd.read_csv(f'ratings_hcs_random1_{pct}pct.csv', index_col=0)
ratings_hcs0_random2 = pd.read_csv(f'ratings_hcs_random2_{pct}pct.csv', index_col=0)
ratings_hcs0_random3 = pd.read_csv(f'ratings_hcs_random3_{pct}pct.csv', index_col=0)

users = pd.read_csv(f'cond_sims_{pct}pct.csv', index_col=0)
items = pd.read_csv(f'drug_sims_{pct}pct.csv', index_col=0)

hcs_0 = datasets.Dataset(ratings=ratings_hcs0, users=users, items=items, 
                                    same_item_user_features=False, name="HCS_0")

hcs_1 = datasets.Dataset(ratings=ratings_hcs1, users=users, items=items, 
                                    same_item_user_features=False, name="HCS_1")

hcs_random1 = datasets.Dataset(ratings=ratings_hcs_random1, users=users, items=items, 
                                    same_item_user_features=False, name="HCS_RANDOM1")

hcs_random2 = datasets.Dataset(ratings=ratings_hcs0_random2, users=users, items=items, 
                                    same_item_user_features=False, name="HCS_RANDOM2")

hcs_random3 = datasets.Dataset(ratings=ratings_hcs0_random3, users=users, items=items, 
                                    same_item_user_features=False, name="HCS_RANDOM3")

ct = open_json('clinical_trials.json')

ct_pairs = set()
ct_pairs12 = set()
ct_pairs34 = set()

for k, v in ct.items():
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

# print(len(ct_pairs12)) # 13604
# print(len(ct_pairs34)) # 11255

drugs = open_json("drugs_with_conditions_final.json")

orig_pairs = set()
for k, v in drugs.items():
    conditions = v['treats']
    for c in conditions:
        orig_pairs.add((c, k))

count = 0
for p in ct_pairs12.copy():
    if p in orig_pairs:
        ct_pairs12.remove(p)
        count += 1
# print(count) # 1364 pairs from the external dataset (phases 1/2) appear in the original dataset
# print(len(ct_pairs12)) # 12240

count = 0
for p in ct_pairs34.copy():
    if p in orig_pairs:
        ct_pairs34.remove(p)
        count += 1
# print(count) # 1681 pairs from the external dataset (phases 3/4) appear in the original dataset
# print(len(ct_pairs34)) # 9574

drugs = set()
conds = set()

for (a, b) in ct_pairs12:
    drugs.add(b)
    conds.add(a)

# print(len(drugs)) # 623
# print(len(conds)) # 590

drugs = set()
conds = set()

for (a, b) in ct_pairs34:
    drugs.add(b)
    conds.add(a)

# print(len(drugs)) # 636
# print(len(conds)) # 613

from src.models.training_testing import build_dataset_from_pairs, load_precomputed_pairs, reconstruct_datasets_from_pairs, \
create_coo_matrix, unique_metapath_pairs, unrelated_pairs

ct12 = build_dataset_from_pairs(list(ct_pairs12), users, items, 'CT12', same_item_user_features=False)
ct34 = build_dataset_from_pairs(list(ct_pairs34), users, items, 'CT34', same_item_user_features=False)

def run_pipeline(model_class, reconstructed_sets, params=None, n_experiments=10, ct=None):
    """
    Run experiments using precomputed and reconstructed train/test datasets.
    
    Parameters:
        model_class: a class (not instance) that implements .fit() and .predict_proba()
        reconstructed_sets: output from `reconstruct_datasets_from_pairs(...)`
        params: parameters passed to the model
        n_experiments: number of experiments to run
    """
    results = []
    model_name = model_class.__name__
    if ct:
        print(f"\n========== HCS (CLINICAL TRIAL TEST SET: PHASES {ct[0]} and {ct[1]}) ==========")
        output_file = f"results_{model_name}_{n_experiments}_ct{ct}.csv"
    else:
        print(f"========== HCS (ORIGINAL DATA) ==========")
        output_file = f"results_{model_name}_{n_experiments}_{pct}pct.csv"
        # output_file = f"results_{model_name}_{n_experiments}_.csv"


    for i in range(n_experiments):
        print(f"\n========== EXPERIMENT {i+1} ==========")
        random.seed(i)
        np.random.seed(i)

        test_dataset = reconstructed_sets[i]['test_dataset']
        train_datasets = reconstructed_sets[i]['train_datasets']

        for name, train_dataset in train_datasets.items():
            print(f"\n--------------- {name.upper()} ---------------")
            start = time()

            # Initialize and fit the model
            print("Fitting model...")
            model = model_class(params)
            model.fit(train_dataset, seed=i)

            # Make predictions
            print("Predicting...")
            scores = model.predict_proba(test_dataset)
            predictions = model.predict(scores)


            # Evaluate
            print("Computing metrics...")
            metrics, _ = validation.compute_metrics(
                scores, predictions, test_dataset,
                metrics=validation.metrics_list, k=10, beta=1, verbose=False
            )

            duration = time() - start

            # Save metrics
            row = {
                'experiment': i + 1,
                'dataset': name,
                'time_seconds': duration
            }
            for metric_name, values in metrics.iterrows():
                row[f'{metric_name}_avg'] = values['Average']
                row[f'{metric_name}_std'] = values['StandardDeviation']

            results.append(row)

    fieldnames = results[0].keys()
    with open(output_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Results saved to {output_file}")

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

f = 'graph.hetmat/path-counts/dwpc_results/sig_dwpc_results_path_scores.tsv'
m0 = build_dataset_from_pairs(list(unique_metapath_pairs(f, list(n_metapaths.keys())[0])), users, items, list(n_metapaths.keys())[0], same_item_user_features=False)
m1 = build_dataset_from_pairs(list(unique_metapath_pairs(f, list(n_metapaths.keys())[1])), users, items, list(n_metapaths.keys())[1], same_item_user_features=False)
m2 = build_dataset_from_pairs(list(unique_metapath_pairs(f, list(n_metapaths.keys())[2])), users, items, list(n_metapaths.keys())[2], same_item_user_features=False)
m3 = build_dataset_from_pairs(list(unique_metapath_pairs(f, list(n_metapaths.keys())[3])), users, items, list(n_metapaths.keys())[3], same_item_user_features=False)
m4 = build_dataset_from_pairs(list(unique_metapath_pairs(f, list(n_metapaths.keys())[4])), users, items, list(n_metapaths.keys())[4], same_item_user_features=False)
m5 = build_dataset_from_pairs(list(unique_metapath_pairs(f, list(n_metapaths.keys())[5])), users, items, list(n_metapaths.keys())[5], same_item_user_features=False)
m6 = build_dataset_from_pairs(list(unique_metapath_pairs(f, list(n_metapaths.keys())[6])), users, items, list(n_metapaths.keys())[6], same_item_user_features=False)
m7 = build_dataset_from_pairs(list(unique_metapath_pairs(f, list(n_metapaths.keys())[7])), users, items, list(n_metapaths.keys())[7], same_item_user_features=False)
m8 = build_dataset_from_pairs(list(unique_metapath_pairs(f, list(n_metapaths.keys())[8])), users, items, list(n_metapaths.keys())[8], same_item_user_features=False)
m9 = build_dataset_from_pairs(list(unique_metapath_pairs(f, list(n_metapaths.keys())[9])), users, items, list(n_metapaths.keys())[9], same_item_user_features=False)

saved_sets = load_precomputed_pairs(f"train_test_pairs_{pct}pct100.json") # pruned graphs
saved_sets = load_precomputed_pairs(f"train_test_pairs100.json") # complete graph
saved_sets12 = load_precomputed_pairs(f"train_test_pairs_ct12_100.json") # clinical trials phases I/II
saved_sets34 = load_precomputed_pairs(f"train_test_pairs_ct34_100.json") # clinical trials phases III/IV
saved_sets_mps = load_precomputed_pairs(f"train_test_pairs_mps_all.json") # importance of metapaths

# comment out ct12 and ct34 for complete and pruned graphs
dataset_lookup = {
    'CT12': ct12,
    'CT34': ct34,
    'HCS_0': hcs_0,
    'HCS_1': hcs_1,
    'HCS_RANDOM1': hcs_random1,
    'HCS_RANDOM2': hcs_random2,
    'HCS_RANDOM3': hcs_random3,
}

dataset_lookup_mps = {
    'HCS_0': hcs_0,
    'HCS_1': hcs_1,
    'Dt>PRa>CO': m0,
    'Dh>CA<hDi>CO': m1,
    'Dh>CH<hDi>CO': m2,
    'Di>CO<aPRa>CO': m3,
    'Dh>PH<hDi>CO': m4,
    'Dh>T<hDi>CO': m5,
    'Dt>PR<tDi>CO': m6,
    'Dt>PRh>B<hPRa>CO': m7,
    'Dt>PRh>CE<hPRa>CO': m8,
    'Dt>PRh>M<hPRa>CO': m9,
}

reconstructed_sets = reconstruct_datasets_from_pairs(
    'HCS_0',
    saved_sets,
    dataset_lookup=dataset_lookup,
    shape=hcs_0.ratings.shape,
    create_coo_matrix=create_coo_matrix
)

reconstructed_sets12 = reconstruct_datasets_from_pairs(
    'CT12',
    saved_sets12,
    dataset_lookup=dataset_lookup,
    shape=hcs_0.ratings.shape,
    create_coo_matrix=create_coo_matrix
)

reconstructed_sets34 = reconstruct_datasets_from_pairs(
    'CT34',
    saved_sets34,
    dataset_lookup=dataset_lookup,
    shape=hcs_0.ratings.shape,
    create_coo_matrix=create_coo_matrix
)

reconstructed_sets_mps = reconstruct_datasets_from_pairs(
    'HCS_0',
    saved_sets_mps,
    dataset_lookup=dataset_lookup_mps,
    shape=hcs_0.ratings.shape,
    create_coo_matrix=create_coo_matrix
)

run_pipeline(models.ALSWR, reconstructed_sets=reconstructed_sets, n_experiments=100)
run_pipeline(models.PMF, reconstructed_sets=reconstructed_sets, n_experiments=100)
run_pipeline(models.LogisticMF, reconstructed_sets=reconstructed_sets, n_experiments=100)
run_pipeline(models.BNNR, reconstructed_sets=reconstructed_sets, n_experiments=5)

run_pipeline(models.ALSWR, reconstructed_sets=reconstructed_sets12, n_experiments=100, ct='12')
run_pipeline(models.PMF, reconstructed_sets=reconstructed_sets12, n_experiments=100, ct='12')
run_pipeline(models.LogisticMF, reconstructed_sets=reconstructed_sets12, n_experiments=100, ct='12')
run_pipeline(models.BNNR, reconstructed_sets=reconstructed_sets12, n_experiments=5, ct='12')

run_pipeline(models.ALSWR, reconstructed_sets=reconstructed_sets34, n_experiments=100, ct='34')
run_pipeline(models.PMF, reconstructed_sets=reconstructed_sets34, n_experiments=100, ct='34')
run_pipeline(models.LogisticMF, reconstructed_sets=reconstructed_sets34, n_experiments=100, ct='34')
run_pipeline(models.BNNR, reconstructed_sets=reconstructed_sets34, n_experiments=5, ct='34')