import numpy as np
import pandas as pd
import copy
from utils import open_json
import csv

drugs = open_json('drugs_with_conditions_final.json')
conditions = open_json('conditions.json')

drug_dict = {drug: [] for drug in drugs.keys()}
cond_dict = {cond: copy.deepcopy(info['slim_terms']) for cond, info in conditions.items()}

rels = (
    'HAS_THERAPEUTIC_GROUP', 
    'HAS_PHARMACOLOGICAL_GROUP', 
    'HAS_CHEMICAL_GROUP', 
    'TARGETS', 
    'HAS_CATEGORY', 
    'ASSOCIATED_WITH'
)

pct = 20 # for pruned graphs (remove pct from the file names for full graph)

with open(f'graph_triples_{pct}pct.txt', 'r') as f:
    for line in f:
        line = line.strip()
        source, rel, target = line.split()
        
        if rel not in rels:
            continue
        
        if source in drug_dict:
            drug_dict[source].append(target)
            
        if target in cond_dict:
            cond_dict[target].append(source)
            
count1 = 0
count2 = 0
for k, v in drugs.items():
    if len(v['treats']) <= 1:
        count1 += 1
    if len(v['treats']) <= 2:
        count2 += 1
print(count1, count2)
print(count1/len(drugs), count2/len(drugs)) # percentage drugs with less than or equal to 1 or 2 conditions

cond_list = list(cond_dict.keys())
n = len(cond_list)
similarity_matrix_cond = np.zeros((n, n))

for i in range(n):
    for j in range(i, n):
        cond_a = cond_list[i]
        cond_b = cond_list[j]
        
        features_a = set(cond_dict[cond_a])
        features_b = set(cond_dict[cond_b])
        
        intersection = features_a & features_b
        union = features_a | features_b
        
        similarity = len(intersection) / len(union) if union else 0.0
        similarity_matrix_cond[i][j] = similarity
        similarity_matrix_cond[j][i] = similarity
        
cond_df = pd.DataFrame(similarity_matrix_cond, index=cond_list, columns=cond_list)
cond_df = cond_df.sort_index(axis=0).sort_index(axis=1)
cond_df.to_csv(f'cond_sims_{pct}pct.csv', index=True)

drug_list = list(drug_dict.keys())
n = len(drug_list)
similarity_matrix_drug = np.zeros((n, n))

for i in range(n):
    for j in range(i, n):
        drug_a = drug_list[i]
        drug_b = drug_list[j]
        
        features_a = set(drug_dict[drug_a])
        features_b = set(drug_dict[drug_b])
        
        intersection = features_a & features_b
        union = features_a | features_b
        
        similarity = len(intersection) / len(union) if union else 0.0
        similarity_matrix_drug[i][j] = similarity
        similarity_matrix_drug[j][i] = similarity
        
drug_df = pd.DataFrame(similarity_matrix_drug, index=drug_list, columns=drug_list)
drug_df = drug_df.sort_index(axis=0).sort_index(axis=1)
drug_df.to_csv(f'drug_sims_{pct}pct.csv', index=True)

print((drug_df == 0).sum().sum()/(len(drug_df))**2) # percentage of zeros
print((cond_df == 0).sum().sum()/(len(cond_df))**2)

# HCS_0
associations = {}

with open(f'graph_triples_{pct}pct.txt', 'r') as f:
    for line in f:
        line = line.strip()
        source, rel, target = line.split()
        
        if rel != 'INDICATED_FOR':
            continue
            
        associations[(source, target)] = 1

sources = sorted(set(src for src, tgt in associations.keys()))
targets = sorted(set(tgt for src, tgt in associations.keys()))

ratings = pd.DataFrame(0, index=sources, columns=targets)

for (src, tgt), val in associations.items():
    ratings.at[src, tgt] = val

ratings.to_csv(f'ratings_hcs0_{pct}pct.csv', index=True)

input_path = f'/workspace/hcs/graph.hetmat{pct}/path-counts/dwpc_results/all_dwpc_results_with_p.tsv'

# HCS_1
with open(input_path, 'r', newline='', encoding='utf-8') as tsvfile:
    reader = csv.DictReader(tsvfile, delimiter='\t')
    for row in reader:
        source_id = row['source_id']
        target_id = row['target_id']
        adj_p = float(row['adjusted_p_value'])

        key = (source_id, target_id)

        if adj_p <= 0.05 and key not in associations:
            associations[key] = 1

sources = sorted(set(src for src, tgt in associations.keys()))
targets = sorted(set(tgt for src, tgt in associations.keys()))

df_associations = pd.DataFrame(0, index=sources, columns=targets)

for (src, tgt), val in associations.items():
    df_associations.at[src, tgt] = val

df_associations.to_csv(f'ratings_hcs1_{pct}pct.csv', index=True)

# HCS_RANDOM(1,2,3)
def generate_random_hcs_matrices(n, original, hcs):
    for i in range(n):
        ratings_hcs = pd.read_csv(hcs, index_col=0)
        ones_count_dict = ((ratings_hcs == 1).sum(axis=0)).to_dict()

        ratings = pd.read_csv(original, index_col=0)

        np.random.seed(i)

        for col in ratings.columns:
            if col in ones_count_dict:
                target_ones = ones_count_dict[col]
                current_ones = (ratings[col] == 1).sum()
                num_to_add = target_ones - current_ones

                if num_to_add > 0:
                    available_rows = ratings.index[ratings[col] != 1]
                    selected_rows = np.random.choice(
                        available_rows,
                        size=min(num_to_add, len(available_rows)),
                        replace=False
                    )
                    ratings.loc[selected_rows, col] = 1

        ratings.to_csv(f'ratings_hcs_random{i+1}_{pct}pct.csv')

generate_random_hcs_matrices(3, f'ratings_hcs0_{pct}pct.csv', f'ratings_hcs1_{pct}pct.csv')