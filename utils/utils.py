import json
import re

def open_json(filename):
    with open(filename, 'r') as json_file:
        data = json.load(json_file)
    return data

def save_json(data, filename):
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    print(f"{filename} successfully saved.")

def clean_text(text):
    # Remove citations like [L41539,A3] or [L41539]
    text = re.sub(r"\[[A-Z][A-Za-z0-9, ]+\]", "", text)  
    # Remove citations like (PubMed:2019570, PubMed:21976677)
    text = re.sub(r"\((?:PubMed:\w+(?:, )?)+\)", "", text)  
    # Remove newline and carriage return characters
    text = text.replace("\n", " ").replace("\r", " ")  
    return text