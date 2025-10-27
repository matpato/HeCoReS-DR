import pandas as pd
from utils.utils import open_json, save_json
from collections import defaultdict

genes = pd.read_csv('CTD_genes.csv')
proteins = open_json('proteins.json')

geneprot_map = {}
for _, row in genes.iterrows():
    ID = row['GeneID']
    if pd.notna(row['UniProtIDs']):
        uniprots = row['UniProtIDs'].split('|')
        uniprots = [uniprot for uniprot in uniprots if uniprot in proteins]
        if len(uniprots) != 0:
            geneprot_map[ID] = uniprots

condgenes = pd.read_csv('/data/processed/CTD_curated_genes_diseases.csv')

gene_cond_map = defaultdict(list)
for _, row in condgenes.iterrows():
    gene_cond_map[row['GeneID']].append(row['DiseaseID'])

protein_cond_map = defaultdict(list)

for gene, conditions in gene_cond_map.items():
    if gene in geneprot_map:  # Check if the gene has associated proteins
        for protein in geneprot_map[gene]:  # Get proteins linked to the gene
            protein_cond_map[protein].extend(conditions)  # Link them to conditions

for protein in protein_cond_map:
    protein_cond_map[protein] = list(set(protein_cond_map[protein]))

for prot in proteins.keys():
    if prot in protein_cond_map:
        proteins[prot]['conditions'] = protein_cond_map[prot]
    else:
        proteins[prot]['conditions'] = []

save_json(proteins, 'proteins.json')
