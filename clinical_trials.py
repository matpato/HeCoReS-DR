import xml.etree.ElementTree as ET
import zipfile
from utils import open_json, save_json

def zip_reader(path, max_records=None):
    """
    Yield (filename, parsed XML tree) for each XML file inside the ClinicalTrials.gov zip.
    """
    with zipfile.ZipFile(path) as open_zip:
        filenames = [f for f in open_zip.namelist() if f.endswith(".xml") and not f.endswith("/")]
        for i, filename in enumerate(filenames):
            with open_zip.open(filename) as open_xml:
                yield filename, ET.parse(open_xml)
            if max_records is not None and i + 1 >= max_records:
                break

def extract_fields(tree):
    """
    Extract relevant fields from a parsed XML tree of a clinical study.
    """
    root = tree.getroot()

    def get_text(path):
        elem = root.find(path)
        return elem.text.strip() if elem is not None and elem.text else None

    def get_all_texts(path):
        return [el.text.strip() for el in root.findall(path) if el.text]

    return {
        "nct_id": get_text("id_info/nct_id"),
        "brief_title": get_text("brief_title"),
        "official_title": get_text("official_title"),
        "brief_summary": get_text("brief_summary"),
        "detailed_description": get_text("detailed_description"),
        "overall_status": get_text("overall_status"),
        "start_date": get_text("start_date"),
        "completion_date": get_text("completion_date"),
        "phase": get_text("phase"),
        "study_type": get_text("study_type"),
        "condition_mesh_terms": get_all_texts("condition"),
        "intervention_mesh_terms": get_all_texts("intervention_browse/mesh_term"),
    }

def load_all_studies(zip_path, max_records=None):
    """
    Returns a dictionary of all extracted studies, keyed by nct_id.
    """
    studies = {}
    for filename, tree in zip_reader(zip_path, max_records=max_records):
        data = extract_fields(tree)
        nct_id = data["nct_id"]
        if nct_id:
            studies[nct_id] = data
    return studies

all_studies = load_all_studies("ctg-public-xml.zip")
save_json(all_studies, "clinical_trials.json")

conditions = open_json("conditions.json")
drugs = open_json("drugs_with_conditions_final.json")

def get_condition_id(cond):
    for ID, condition in conditions.items():
        if condition['name'].lower() == cond.lower() or any(syn.lower() == cond.lower() for syn in condition['synonyms']):
            return ID
        
def conditions_to_ids(conds):
    return [get_condition_id(cond) for cond in conds]

def get_drug_id(drug):
    for k, v in drugs.items():
        if v['name'] == drug or drug in v['synonyms']:
            return k
        
for k, v in all_studies.items():
    conds = v['condition_mesh_terms']
    interventions = v['intervention_mesh_terms']

    # Map and filter conditions
    condition_ids = []
    valid_conds = []
    for cond in conds:
        cid = get_condition_id(cond)
        if cid is not None:
            condition_ids.append(cid)
            valid_conds.append(cond)
    
    # Map and filter interventions
    drug_ids = []
    valid_drugs = []
    for drug in interventions:
        did = get_drug_id(drug)
        if did is not None:
            drug_ids.append(did)
            valid_drugs.append(drug)

    # Update the dictionary
    v['conditions'] = condition_ids
    v['drugs'] = drug_ids
    v['condition_mesh_terms'] = valid_conds
    v['intervention_mesh_terms'] = valid_drugs

ct = {k: v for k, v in all_studies.items() if v['conditions'] and v['drugs']}
    
save_json(ct, "clinical_trials.json")