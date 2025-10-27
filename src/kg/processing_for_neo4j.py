from utils.utils import save_json, open_json
from collections import defaultdict
from copy import deepcopy

# Futher processing
drugs = open_json('drugs_with_conditions.json')
conditions = open_json('conditions.json')
conditions_no_slims = open_json('conditions_no_slims.json')
proteins = open_json('proteins.json')
targets = open_json('targets.json')

print(len(drugs))

for drug in ['DB04923', 'DB06361', 'DB09277', 'DB09326', 'DB09332', 'DB09430', 'DB09479', 'DB09493', 'DB09513', 'DB09546', 'DB09563', 
             'DB09571', 'DB10771', 'DB11051', 'DB11056', 'DB11164', 'DB11180', 'DB11200', 'DB11331', 'DB14100', 'DB14478', 'DB14550', 
             'DB15507', 'DB15709', 'DB16783', 'DB17743', 'DB17961']:
    del drugs[drug]

print(len(drugs))

### Drugs with no contextual information

# # DB04923 rhThrombin
# # DB06361 rsPSMA Vaccine
# # DB09277 Choline C 11
# # DB09326 Ammonia N-13
# # DB09332 Kappadione
# # DB09430 Albumin iodinated I-131 serum
# # DB09479 Rubidium Rb-82
# # DB09493 Technetium Tc-99m red blood cells
# # DB09513 Urea C-14
# # DB09546 Iobenguane sulfate I-123
# # DB09563 Choline C-11
# # DB09571 Levmetamfetamine
# # DB10771 Bovine type I collagen
# # DB11051 Azficel-T
# # DB11056 Stannous chloride
# # DB11164 Bicisate
# # DB11180 Tetrofosmin
# # DB11200 Aluminum zirconium octachlorohydrex gly
# # DB11331 1-Palmitoyl-2-oleoyl-sn-glycero-3-(phospho-rac-(1-glycerol))
# # DB14100 Pork collagen
# # DB14478 Povidone K30
# # DB14550 Gallium chloride Ga-67
# # DB15507 Influenza B virus B/Singapore/INFTT-16-0610/2016 antigen (MDCK cell derived, propiolactone inactivated)
# # DB15709 Influenza B virus B/Darwin/7/2019 antigen (MDCK cell derived, propiolactone inactivated)
# # DB16783 Fidanacogene elaparvovec
# # DB17743 Fecal microbiota
# # DB17961 Donislecel

drugs = {drug: info for drug, info in drugs.items() if 'approved' in info['groups']}
print(f"There are {len(drugs)} approved drugs.")

for drug, info in drugs.items():
    treats = [cond for cond in info['treats'] if cond not in conditions_no_slims]
    not_treats = [cond for cond in info['not_treats'] if cond not in conditions_no_slims]

    info['treats'] = treats
    info['not_treats'] = not_treats

# Conditions with associated drugs
conds = set()
for drug, info in drugs.items():
    conds.update(info['treats'])
    
print(f"{len(conds)} conditions associated with drugs.")

# Number of drugs associated with each condition
cond_counts = defaultdict(int)

for drug in drugs.values():
    for cond in drug['treats']:
        cond_counts[cond] += 1

cond_counts = dict(sorted(cond_counts.items(), key=lambda x: x[1], reverse=True))

# Number of drugs associated with each condition (name)
cond_name_counts = {}
for cond, info in cond_counts.items():
    cond_name_counts[conditions[cond]['name']] = info

# Conditions not specific enough or that are not applicable (e.g. Death)

invalid_conds = {
 'Neoplasms': 112,
 'Pain': 108,
 'Infections': 101,
 'Inflammation': 71,
 'Drug Hypersensitivity': 46,
 'Wounds and Injuries': 43,
 'Bacterial Infections': 39,
 'Skin Diseases': 34,
 'Kidney Diseases': 28,
 'Cardiovascular Diseases': 22,
 'Skin Diseases, Infectious': 21,
 'Neoplasm Metastasis': 21,
 'Poisoning': 17,
 'Gastrointestinal Diseases': 17,
 'Psychotic Disorders': 17,
 'Drug-Related Side Effects and Adverse Reactions': 16,
 'Medically Unexplained Symptoms': 16,
 'Migraine Disorders': 15,
 'Alcoholism': 13,
 'Mental Disorders': 13,
 'Rheumatic Diseases': 12,
 'Acute Pain': 11,
 'Endocrine System Diseases': 11,
 'Liver Diseases': 10,
 'Chronic Pain': 10,
 'Disease': 10,
 'Growth Disorders': 9,
 'Respiratory Tract Diseases': 9,
 'Parkinsonian Disorders': 9,
 'Skin Diseases, Bacterial': 9,
 'Myelodysplastic Syndromes': 9,
 'Death': 8,
 'Substance-Related Disorders': 8,
 'Acute Disease': 7,
 'Bone Diseases': 7,
 'Nervous System Diseases': 7,
 'Peripheral Vascular Diseases': 7,
 'Immune System Diseases': 7,
 'Basal Ganglia Diseases': 6,
 'Mood Disorders': 6,
 'Sneezing': 6,
 'Deficiency Diseases': 6,
 'Neuroendocrine Tumors': 6,
 'Spinal Cord Injuries': 5,
 'Malabsorption Syndromes': 5,
 'Carotid Artery Diseases': 5,
 'Bone Diseases, Infectious': 5,
 'Genetic Diseases, Inborn': 5,
 'Stomach Diseases': 5,
 'Eye Diseases': 5,
 'Bone Diseases, Metabolic': 5,
 'Heart Diseases': 5,
 'Hematologic Diseases': 5,
 'Hemorrhagic Disorders': 4,
 'Connective Tissue Diseases': 4,
 'Heart Defects, Congenital': 4,
 'Brain Diseases': 4,
 'Movement Disorders': 4,
 'Lung Diseases': 4,
 'Airway Obstruction': 4,
 'Cardiomyopathies': 4,
 'Feeding and Eating Disorders': 4,
 'Vision Disorders': 4,
 'Mouth Diseases': 4,
 'Bone Marrow Diseases': 3,
 'Motor Neuron Disease': 3,
 'Hypothalamic Diseases': 3,
 'Ovarian Diseases': 3,
 'Pancreatic Diseases': 3,
 'Peripheral Nervous System Diseases': 3,
 'Respiratory Sounds': 3,
 'Thyroid Diseases': 3,
 'Opioid-Related Disorders': 3,
 'Wound Infection': 3,
 'Lacrimal Apparatus Diseases': 3,
 'Sexual Dysfunction, Physiological': 3,
 'Lung Diseases, Interstitial': 3,
 'Disruptive, Impulse Control, and Conduct Disorders': 3,
 'Cerebrovascular Disorders': 3,
 'Musculoskeletal Diseases': 3,
 'Urinary Bladder Diseases': 3,
 'Virus Diseases': 3,
 'Carcinoma': 3,
 'Chronic Disease': 3,
 'Blood Coagulation Disorders': 3,
 'Developmental Disabilities': 3,
 'Postoperative Complications': 3,
 'Metabolic Diseases': 3,
 'Rectal Diseases': 3,
'Eye Infections': 2,
'Primary Immunodeficiency Diseases': 2,
'Food Hypersensitivity': 2,
'Flushing': 2,
'Carcinoid Tumor': 2,
'Avitaminosis': 2,
'Chemical and Drug Induced Liver Injury': 2,
'Central Nervous System Diseases': 2,
'Pelvic Inflammatory Disease': 2,
'Biliary Tract Diseases': 2,
'Community-Acquired Infections': 2,
'Abnormalities, Drug-Induced': 2,
'Tick-Borne Diseases': 2,
'Sexually Transmitted Diseases': 2,
'Premature Birth': 2,
'Abortion, Spontaneous': 2,
'Somatoform Disorders': 2,
'Heart Valve Diseases': 2,
'Neurologic Manifestations': 2,
'Gram-Negative Bacterial Infections': 2,
'Signs and Symptoms, Digestive': 2,
'Urogenital Diseases': 2,
'Vestibular Diseases': 2,
'Vaginal Diseases': 2,
'Myelodysplastic-Myeloproliferative Diseases': 2,
'Blood Platelet Disorders': 2,
'Soft Tissue Injuries': 2,
'Urologic Diseases': 2,
'Fetal Death': 2,
'Soft Tissue Infections': 2,
'Bronchial Diseases': 2,   
'Intellectual Disability': 2,
'Muscular Diseases': 2,
'Congenital Abnormalities': 2,
'Vascular Diseases': 2,
'Gallbladder Diseases': 2,
'Urea Cycle Disorders, Inborn': 2,
'Neurodegenerative Diseases': 2,
'Coronaviridae Infections': 2,
'Cartilage Diseases': 2,
'Metabolic Syndrome': 2,
'Carcinoma in Situ': 2,
'Skin Neoplasms': 2,
'Syndrome': 2,
'Testicular Diseases': 2
# 'Pituitary Diseases': 1,
# 'Headache Disorders': 1,
# 'Malnutrition': 1,
# 'Precancerous Conditions': 1,
# 'Spinal Cord Diseases': 1,
# 'Brain Injuries, Traumatic': 1,
# 'Genital Diseases': 1,  
# 'Herpesviridae Infections': 1,
# 'Genital Diseases, Female': 1,
# 'Pre-Excitation Syndromes': 1,
# 'Carcinoma, Squamous Cell': 1,
# 'Pelvic Infection': 1,
# 'Parathyroid Diseases': 1,
# 'Cognition Disorders': 1,
# 'Critical Illness': 1,
# 'Olfaction Disorders': 1,
# 'Central Nervous System Infections': 1,
# 'Corneal Injuries': 1,
# 'Chromosome Aberrations': 1,
# 'Nail Diseases': 1,
# 'Metabolism, Inborn Errors': 1,
# 'Epilepsy, Absence': 1,
# 'Neglected Diseases': 1,
# 'Vulvar Diseases': 1,
# 'Brain Injuries': 1,
# 'Gram-Positive Bacterial Infections': 1,
# 'Personality Disorders': 1,
# 'Skin and Connective Tissue Diseases': 1,
# 'Obstetric Labor, Premature': 1,
# 'Abortion, Missed': 1,
# 'Neurocognitive Disorders': 1,
# 'Chronobiology Disorders': 1,
# 'Sleep Disorders, Circadian Rhythm': 1,
# 'Signs and Symptoms, Respiratory': 1,
# 'Bites and Stings': 1,
# 'Phobic Disorders': 1,
# 'Urethral Diseases': 1,
# 'Invasive Fungal Infections': 1,
# 'Visceral Pain': 1,
# 'Communicable Diseases': 1,
# 'Labyrinth Diseases': 1,
# 'Nociceptive Pain': 1,
# 'Binge-Eating Disorder': 1,
# 'Immune Deficiency Disease': 1,
# 'Nutritional and Metabolic Diseases': 1,
# 'Bile Duct Diseases': 1,
# 'Peroxisomal Disorders': 1,
# 'Porphyrias': 1,
# 'Sleep-Wake Transition Disorders': 1,
# 'Tic Disorders': 1,
# 'Sexual Dysfunctions, Psychological': 1,
# 'Breast Diseases': 1,
# 'AIDS-Related Opportunistic Infections': 1,
# 'Autoimmune Diseases': 1,
# 'Hereditary Autoinflammatory Diseases': 1,
# 'Drug Misuse': 1,
# 'Amyloid Neuropathies, Familial': 1,
# 'Iatrogenic Disease': 1,
# 'Laminopathies': 1,
# 'Aortic Diseases': 1,
# 'Photosensitivity Disorders': 1,
# 'Fractures, Spontaneous': 1,
# 'Facial Asymmetry': 1,
# 'Infertility, Male': 1,
# 'Spondylarthropathies': 1,
# 'Lung Diseases, Obstructive': 1,
# 'Tongue Diseases': 1,
# 'Fractures, Stress': 1,
# 'Paranoid Disorders': 1,
# 'Joint Diseases': 1,
# 'Neurotoxicity Syndromes': 1,
# 'Duodenal Diseases': 1,
# 'Lip Diseases': 1,
# 'Optic Nerve Diseases': 1,
# 'Gestational Weight Gain': 1,
# 'Intestinal Diseases': 1,
# 'Battered Child Syndrome': 1,
# 'Lacerations': 1,
# 'Surgical Wound': 1,
# 'Diverticular Diseases': 1,
# 'Neural Tube Defects': 1,
# 'Stomatognathic Diseases': 1,
# 'Tooth Abnormalities': 1,
# 'Respiration Disorders': 1,
# 'Tibial Fractures': 1,
# 'Lipid Metabolism Disorders': 1,
# 'Neurofibromatosis 1': 1,
# 'Neurofibroma, Plexiform': 1,
# 'Achondroplasia': 1,
# 'Lymphatic Metastasis': 1,
# 'Chagas Disease': 1,
# 'Hyperplasia': 1,
# 'Urination Disorders': 1,
# 'Catheter-Related Infections': 1,
# 'Niemann-Pick Diseases': 1,
# 'Myasthenic Syndromes, Congenital': 1,
# 'Hemostatic Disorders': 1,
# 'Protozoan Infections': 1,
# 'Parasitic Diseases': 1,
# 'Corneal Diseases': 1,
# 'Lactose Intolerance': 1,
# 'Carcinogenesis': 1,
# 'Blood Coagulation Disorders, Inherited': 1,
# 'IMMUNE SUPPRESSION': 1,
# 'Neoplasms, Glandular and Epithelial': 1,
# 'Leber Congenital Amaurosis': 1,
# 'Retinal Degeneration': 1,
# 'Blindness': 1,
# 'Retinal Dystrophies': 1,
# 'Signs and Symptoms': 1,
# 'Ectoparasitic Infestations': 1,
# 'Weight Gain': 1,
# 'Skin Diseases, Eczematous': 1,
# 'Severe Combined Immunodeficiency': 1,
# 'Multiple Organ Failure': 1,
# 'Insulin Resistance': 1,
# 'Porphyrias, Hepatic': 1,
# 'Protein-Losing Enteropathies': 1
}

print(len(invalid_conds))

# percentage of invalid conditions (in the universe of conditions with associated drugs)
print(len(invalid_conds)/len(conds))

def get_condition_id(cond):
    for ID, condition in conditions.items():
        if condition['name'].lower() == cond.lower() or any(syn.lower() == cond.lower() for syn in condition['synonyms']):
            return ID
        
invalid_conds = [get_condition_id(cond) for cond in invalid_conds.keys()]

count = 0
for k, v in drugs.items():
    if v['treats']:
        count+=1
print(count)

for k, v in drugs.items():
    treats = [cond for cond in v['treats'] if cond not in invalid_conds]
    not_treats = [cond for cond in v['not_treats'] if cond not in invalid_conds]
    
    v['treats'] = treats
    v['not_treats'] = not_treats

count = 0
for k, v in drugs.items():
    if v['treats']:
        count+=1
print(count) # drugs after removing invalid conditions

def prune(drugs, min_conditions=2, min_drugs=2):
    drugs = deepcopy(drugs)

    while True:
        # Find invalid drugs (treating fewer than min_conditions)
        invalid_drugs = [d for d, info in drugs.items() if len(info['treats']) < min_conditions]

        # Remove them
        for d in invalid_drugs:
            del drugs[d]

        # Build condition → drugs mapping
        condition_to_drugs = defaultdict(set)
        for d, info in drugs.items():
            for c in info['treats']:
                condition_to_drugs[c].add(d)

        # Find invalid conditions (treated by fewer than min_drugs)
        invalid_conditions = [c for c, ds in condition_to_drugs.items() if len(ds) < min_drugs]

        # Remove invalid conditions from remaining drugs
        for d, info in drugs.items():
            info['treats'] = [c for c in info['treats'] if c not in invalid_conditions]

        # If no removals
        if not invalid_drugs and not invalid_conditions:
            break

    # Surviving conditions = keys in final condition_to_drugs
    surviving_conditions = set(condition_to_drugs.keys())

    return drugs, surviving_conditions

drugs, surviving_conditions = prune(drugs)
print(len(drugs))
print(len(surviving_conditions))

for d, info in drugs.items():
    assert len(info['treats']) >= 2, f"Drug {d} treats fewer than 2 conditions"

condition_to_drugs = defaultdict(set)
for d, info in drugs.items():
    for c in info['treats']:
        condition_to_drugs[c].add(d)

for c, drugs_for_c in condition_to_drugs.items():
    assert len(drugs_for_c) >= 2, f"Condition {c} is treated by fewer than 2 drugs"

conditions = {cond: info for cond, info in conditions.items() if cond in surviving_conditions}
print(len(drugs))
print(len(conditions))

save_json(drugs, 'drugs_with_conditions_final.json')
save_json(conditions, 'conditions.json')

# Remove protein-condition assosciations for invalid conditions

for prot in proteins:
    proteins[prot]['conditions'] = [
        cond for cond in proteins[prot].get('conditions', [])
        if cond in surviving_conditions
    ]

save_json(proteins, 'proteins.json')

# Maintain only drug-protein interactions with a known therapeutic effect
to_delete = []

for drug, info in targets.items():
    if drug in drugs:
        info[:] = [item for item in info if item['known_action'] == "yes"]
    else:
        to_delete.append(drug)

for drug in to_delete:
    del targets[drug]

save_json(targets, 'targets.json')