import pandas as pd
from utils import save_json

ctd = pd.read_csv("CTD_diseases.csv")
ctd = ctd.drop(columns=['TreeNumbers', 'ParentTreeNumbers'])

# Replace occurrences of '|' with '; ' for better readability
ctd[['ParentIDs', 'Synonyms', 'SlimMappings']] = ctd[['ParentIDs', 'Synonyms', 'SlimMappings']].fillna('').apply(
    lambda x: x.str.replace(r'\s*\|\s*', '; ', regex=True)
)

cols = ['AltDiseaseIDs', 'Definition', 'ParentIDs', 'Synonyms', 'SlimMappings']
ctd[cols] = ctd[cols].fillna('')

ctd['Definition'] = ctd['Definition'].fillna('')
ctd.loc[3891, 'ParentIDs'] = ''

# Helper function to split, strip and filter field values
def parse_field(field_value):
    return [item.strip() for item in field_value.split('; ') if item.strip()] if field_value else []

conditions = {}
conditions_no_slims = set()

for _, row in ctd.iterrows():
    name = row['DiseaseName']
    definition = row['Definition']
    parents = parse_field(row['ParentIDs'])
    synonyms = parse_field(row['Synonyms'])
    slims = parse_field(row['SlimMappings'])
    
    if slims:
        conditions_info = {
            'name': name,
            'definition': definition,
            'parents': parents,
            'synonyms': synonyms,
            'slim_terms': slims
        }

        conditions[row['DiseaseID']] = conditions_info
        
    else:
        conditions_no_slims.add(row['DiseaseID'])

save_json(conditions, 'conditions.json')
save_json(list(conditions_no_slims), 'conditions_no_slims.json')

### Conditions with no SLIM terms

# Toxemia
# Nutritional and Metabolic Diseases
# Plant Poisoning
# Chemically-Induced Disorders
# Female Urogenital Diseases and Pregnancy Complications
# Catheter-Related Infections
# Congenital, Hereditary, and Neonatal Diseases and Abnormalities
# Drug Hypersensitivity
# Hemic and Lymphatic Diseases
# Lathyrism
# Sexual Dysfunction, Physiological
# Metabolic Side Effects of Drugs and Substances
# Shellfish Poisoning
# Silhouette sign
# Soft Tissue Infections
# Southern tick-associated rash illness
# Mushroom Poisoning
# Diseases
# Tick Toxicoses
# Vaccine-Preventable Diseases
# Carbon Monoxide Poisoning
# Infertility
# Ergotism
# Tick Paralysis
# Superinfection
# Tick-Borne Diseases
# Heavy Metal Poisoning
# Urogenital Diseases
# Vector Borne Diseases
# Ciguatera Poisoning
# Abdominal Abscess
# Milk Sickness
# Drug-Related Side Effects and Adverse Reactions
# Propofol Infusion Syndrome
# Genital Diseases
# Pathological Conditions, Signs and Symptoms
# Skin and Connective Tissue Diseases
# Mosquito-Borne Diseases
# Carbon Tetrachloride Poisoning
# Organophosphate Poisoning
# Cadmium Poisoning
# Serotonin Syndrome
# Mercury Poisoning
# Morphological and Microscopic Findings
# Mycotoxicosis
# Psoas Abscess
# Aflatoxin Poisoning
# Community-Acquired Infections
# Wound Infection
# Foodborne Diseases
# Squamous Intraepithelial Lesions
# Jamaican vomiting sickness
# Lead Poisoning
# Latent Infection
# Infections
# Opportunistic Infections
# Pelvic Infection
# Fluoride Poisoning
# Gas Poisoning
# Poisoning
# Coinfection
# Focal Infection
# Intraabdominal Infections
# Waterborne Diseases
# Margins of Excision
# Anticholinergic Syndrome