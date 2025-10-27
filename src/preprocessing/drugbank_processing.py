from lxml import etree
from utils.utils import save_json

def get_text(elements, default=""):
    if isinstance(elements, list):
        return elements[0].text.strip() if elements and elements[0].text else default
    return elements.text.strip() if elements is not None and elements.text else default

def parse_drug_info(xml_file):
    tree = etree.parse(xml_file)
    root = tree.getroot()
    
    ns = {'db': 'http://www.drugbank.ca'}
    drug_data = {}

    for drug in root.xpath(".//db:drug", namespaces=ns):
        drug_id = get_text(drug.xpath(".//db:drugbank-id[@primary='true']", namespaces=ns))
        if not drug_id:
            continue

        fields = ["name", "description", "indication", "pharmacodynamics", 
                  "mechanism-of-action", "toxicity", "metabolism", "absorption", 
                  "half-life", "protein-binding", "route-of-elimination", 
                  "volume-of-distribution", "clearance"]
        
        drug_info = {field.replace("-", "_"): get_text(drug.xpath(f".//db:{field}", namespaces=ns)) for field in fields}

        drug_info["groups"] = [get_text(group) for group in drug.xpath(".//db:groups/db:group", namespaces=ns)]
        drug_info["salts"] = [get_text(salt) for salt in drug.xpath(".//db:salts/db:salt/db:name", namespaces=ns)]
        drug_info["synonyms"] = list(set(get_text(synonym) for synonym in drug.xpath(".//db:synonyms/db:synonym", namespaces=ns)))
        drug_info["products"] = list(set(get_text(product) for product in drug.xpath(".//db:products/db:product/db:name", namespaces=ns)))
        
        atc_codes = set()
        for atc in drug.xpath(".//db:atc-codes/db:atc-code", namespaces=ns):
            atc_codes.add(atc.get("code"))
            atc_codes.update(level.get("code") for level in atc.xpath(".//db:level", namespaces=ns))
        drug_info["atc_codes"] = list(atc_codes)

        categories = []
        for category in drug.xpath(".//db:categories/db:category", namespaces=ns):
            mesh_id = get_text(category.xpath(".//db:mesh-id", namespaces=ns))
            if mesh_id:
                category_name = get_text(category.xpath(".//db:category", namespaces=ns))
                categories.append({"category": category_name, "mesh-id": mesh_id})

        drug_info["categories"] = categories

        drug_data[drug_id] = drug_info
    
    return drug_data

# NOT USED IN THE THESIS
def parse_drug_interactions(xml_file):
    tree = etree.parse(xml_file)
    root = tree.getroot()
    
    ns = {'db': 'http://www.drugbank.ca'}
    drug_interactions = {}

    for drug in root.xpath(".//db:drug", namespaces=ns):
        drug_id = get_text(drug.xpath(".//db:drugbank-id[@primary='true']", namespaces=ns))
        if not drug_id:
            continue

        interactions = drug.xpath(".//db:drug-interactions/db:drug-interaction", namespaces=ns)
        if not interactions:
            continue

        drug_interactions[drug_id] = [
            {
                "drugbank_id": get_text(interaction.xpath("db:drugbank-id", namespaces=ns)),
                "description": get_text(interaction.xpath("db:description", namespaces=ns))
            }
            for interaction in interactions
        ]
    
    return drug_interactions

def parse_drug_targets(xml_file):
    tree = etree.parse(xml_file)
    root = tree.getroot()
    
    ns = {'db': 'http://www.drugbank.ca'}
    drug_targets = {}

    for drug in root.xpath(".//db:drug", namespaces=ns):
        drug_id = get_text(drug.xpath(".//db:drugbank-id[@primary='true']", namespaces=ns), default=None)
        if not drug_id:
            continue

        targets = drug.xpath(".//db:targets/db:target", namespaces=ns)
        if not targets:
            continue

        drug_targets[drug_id] = [
            {
                "name": get_text(target.xpath("db:name", namespaces=ns)),
                "actions": [get_text(action) for action in target.xpath("db:actions/db:action", namespaces=ns)],
                "known_action": get_text(target.xpath("db:known-action", namespaces=ns), default="unknown")
            }
            for target in targets
        ]
    
    return drug_targets

def parse_polypeptides(xml_file):
    tree = etree.parse(xml_file)
    root = tree.getroot()

    ns = {"db": "http://www.drugbank.ca"}  
    polypeptide_data = {}

    for polypeptide in root.xpath(".//db:polypeptide", namespaces=ns):
        polypeptide_id = polypeptide.get("id", "").strip()
        if not polypeptide_id:
            continue

        polypeptide_data[polypeptide_id] = {
            "name": get_text(polypeptide.find("db:name", ns)),
            "general_function": get_text(polypeptide.find("db:general-function", ns)),
            "specific_function": get_text(polypeptide.find("db:specific-function", ns)),
            "gene_name": get_text(polypeptide.find("db:gene-name", ns)),
            "cellular_location": get_text(polypeptide.find("db:cellular-location", ns)),
            "organism": get_text(polypeptide.find("db:organism", ns)),
            "go_classifiers": {
                category: [
                    get_text(classifier.find("db:description", ns))
                    for classifier in polypeptide.xpath(f".//db:go-classifier[db:category='{category}']", namespaces=ns)
                    if get_text(classifier.find("db:description", ns))
                ]
                for category in ["component", "function", "process"]
            }
        }

    return polypeptide_data

# ### Commands for downloading DrugBank's database
# # curl -Lfv -o drugbank.zip -u USERNAME:PASSWORD https://go.drugbank.com/releases/5-1-13/downloads/all-full-database
# # unzip drugbank.zip

xml_file = "full database.xml"

drugs = parse_drug_info(xml_file)
save_json(drugs, 'drugs.json')

targets = parse_drug_targets(xml_file)
save_json(targets, 'targets.json')

proteins = parse_polypeptides(xml_file)
save_json(proteins, 'proteins.json')

# Add specific_function to function key
for protein, info in proteins.items():
    specific_function = info['specific_function']
    if specific_function and specific_function not in info['go_classifiers']['function']:
        info['go_classifiers']['function'].append(specific_function)

save_json(proteins, 'proteins.json')

# Map drug categories to IDs and updating drugs.json
mesh_to_category = {}

for drug, info in drugs.items():
    for category_entry in info.get("categories", []):
        mesh_id = category_entry.get("mesh-id")
        category = category_entry.get("category")
        
        mesh_to_category[mesh_id] = category

save_json(mesh_to_category, 'drug_categories.json')

for drug, info in drugs.items():
    info["categories"] = [category["mesh-id"] for category in info["categories"]]

save_json(drugs, 'drugs.json')