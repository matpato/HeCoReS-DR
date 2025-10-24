from flair.models import EntityMentionLinker
from flair.nn import Classifier
from flair.data import Sentence
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt_tab')
from utils import open_json, save_json, clean_text
import torch
from tqdm import tqdm

conditions = open_json('conditions.json')
drugs = open_json('drugs.json')

approved_drugs = [drug for drug, info in drugs.items() if 'approved' in info['groups']]
print(f"There are {len(approved_drugs)} approved drugs.")

# Load models
tagger = Classifier.load("hunflair2")
disease_linker = EntityMentionLinker.load("disease-linker")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tagger.to(device)
disease_linker.to(device)

def get_condition_id(cond):
    for ID, condition in conditions.items():
        if condition['name'].lower() == cond.lower() or any(syn.lower() == cond.lower() for syn in condition['synonyms']):
            return ID

def get_sentences(text):
    sentences = sent_tokenize(clean_text(text))
    return [s for s in sentences if 'not' not in s.lower() and 'contraindicated' not in s.lower()]

def extract_conditions(text, is_drug=True):
    if is_drug:
        sentences = get_sentences(text)
    else:
        sentences = [text]
    
    tags = set()
    
    for sentence in sentences:
        sentence = Sentence(sentence)
        sentence.to(device)
        tagger.predict(sentence)
        disease_linker.predict(sentence)
    
        conds = {cond.metadata['name'] for cond in sentence.get_labels("link") 
                 if not cond.shortstring.split("/")[0].strip('""').isupper()} # If the entity is not an isolated abbreviation
        
        tags.update(conds)

    return tags

def conditions_to_ids(conds):
    return [get_condition_id(cond) for cond in conds]

for drug, info in tqdm(list(drugs.items())):
    treats = []
    not_treats = []

    if drug in approved_drugs:
        treats = conditions_to_ids(extract_conditions(info['indication']))
        not_treats = conditions_to_ids(extract_conditions(info['toxicity']))

    info['treats'] = treats
    info['not_treats'] = not_treats

save_json(drugs, 'drugs_with_conditions.json')