import pronto
from utils import save_json, open_json

proteins = open_json('proteins.json')

protein_map = {}
go_terms = set()

for prot, info in proteins.items():
    protein_map[info['name']] = prot
    
    for go_c in info['go_classifiers']:
        go_terms.update(info['go_classifiers'][go_c])

save_json(protein_map, 'protein_map.json')

ontology = pronto.Ontology("go.obo")
go_dict = {}

for term in ontology.terms():
    go_dict[term.id] = {
        "name": term.name,
        "definition": str(term.definition),
        "namespace": term.namespace,
        "synonyms": [syn.description for syn in term.synonyms],
        "parents": [parent.id for parent in term.superclasses(distance=1) if parent.id != term.id],
        "children": [child.id for child in term.subclasses(distance=1) if child.id != term.id],
    }

go = {}
go_terms = {term.lower() for term in go_terms}

for go_id, info in go_dict.items():
    if info['name'].lower() in go_terms or any(syn.lower() in go_terms for syn in info['synonyms']):
        go[go_id] = {
            'name': info['name'],
            'definition': str(info['definition']),
            'namespace': info['namespace'],
            'synonyms': info['synonyms'],
            'parents': info['parents'],
            'children': info['children'],
        }

go_name_to_id = {}
for go_id, info in go.items():
    go_name_to_id[info['name']] = go_id

save_json(go, 'go.json')
save_json(go_name_to_id, 'go_name_to_id.json')

targets = open_json('targets.json')

# Update targets with 'protein_id'
for drug, target_list in targets.items():
    for target in target_list:
        protein_name = target['name']
        if protein_name in protein_map:
            target['protein_id'] = protein_map[protein_name]

# Remove targets without 'protein_id' for all drugs
for drug, target_list in targets.items():
    targets[drug] = [target for target in target_list if 'protein_id' in target]

save_json(targets, 'targets.json')