import pandas as pd
from utils.utils import save_json, open_json

atc_df = pd.read_csv('/data/processed/WHO ATC-DDD 2024-07-31.csv')
atc_df = atc_df.drop(columns=['ddd', 'uom', 'adm_r', 'note'])
atc_df['atc_name'] = atc_df['atc_name'].str.capitalize()
atc_df = atc_df.drop_duplicates()

atc_codes = list(atc_df['atc_code'])
atc_codes = sorted(atc_codes)

def build_hierarchy(codes):
    hierarchy = {}
    ref = {}

    for code in sorted(codes):
        # Determine the parent by finding the longest existing prefix
        parent = None
        for existing in ref:
            if code.startswith(existing) and existing != code:
                parent = existing

        if parent is None:
            hierarchy[code] = {}
            ref[code] = hierarchy[code]
        else:
            ref[parent][code] = {}
            ref[code] = ref[parent][code]

    return hierarchy

atc_dict = build_hierarchy(atc_codes)

def find_parents(nested_dict, target_key, return_all=False, path=None):
    if path is None:
        path = []
    
    for key, value in nested_dict.items():
        if key == target_key:
            return path if return_all else (path[-1] if path else None)
        if isinstance(value, dict):
            result = find_parents(value, target_key, return_all, path + [key])
            if result is not None:
                return result
    
    return None

atc = {}

for code in atc_codes:
    name = atc_df[atc_df['atc_code'] == code]['atc_name'].tolist()[0]
    parent = find_parents(atc_dict, code)
    len_map = {1: 'anatomical', 3: 'therapeutic', 4: 'pharmacological', 5: 'chemical', 7: 'substance'}

    atc[code] = {
        'name': atc_df[atc_df['atc_code'] == code]['atc_name'].tolist()[0],
        'parent': find_parents(atc_dict, code),
        'group': len_map[len(code)]
    }

save_json(atc, 'atc.json')

drugs_with_conditions = open_json('drugs_with_conditions.json')

def find_elements_with_length(lst, x):
    return [element for element in lst if len(element) == x]

for drug, info in drugs_with_conditions.items():
    atc = info['atc_codes']
    info['anatomical_groups'] = find_elements_with_length(atc, 1)
    info['therapeutic_groups'] = find_elements_with_length(atc, 3)
    info['pharmacological_groups'] = find_elements_with_length(atc, 4)
    info['chemical_groups'] = find_elements_with_length(atc, 5)

save_json(drugs_with_conditions, 'drugs_with_conditions.json')