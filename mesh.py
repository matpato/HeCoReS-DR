import requests
from lxml import etree
from utils import save_json, open_json

# URL of the MeSH descriptors file
url = "https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc2025.gz"

# Download the file and save it directly as an XML file
response = requests.get(url)
with open("desc2025.xml", "wb") as f:
    f.write(response.content)

# Parse the XML file
tree = etree.parse("desc2025.xml")
root = tree.getroot()

# Dictionary to map TreeNumbers to DescriptorUI
tree_to_ui = {}

# Dictionary to store the final structured data (without TreeNumbers)
mesh_hierarchy = {}

# Build mapping of TreeNumbers to DescriptorUI
for descriptor in root.xpath("//DescriptorRecord"):
    descriptor_ui = descriptor.findtext("DescriptorUI")
    tree_numbers = [tn.text for tn in descriptor.xpath(".//TreeNumberList/TreeNumber")]

    # Store each TreeNumber mapped to its DescriptorUI
    for tn in tree_numbers:
        tree_to_ui[tn] = descriptor_ui

# Extract required information
for descriptor in root.xpath("//DescriptorRecord"):
    descriptor_ui = descriptor.findtext("DescriptorUI")
    descriptor_name = descriptor.findtext("DescriptorName/String")
    scope_note = descriptor.findtext(".//ScopeNote")
    tree_numbers = [tn.text for tn in descriptor.xpath(".//TreeNumberList/TreeNumber")]

    parent_descriptor_uis = set()

    for tn in tree_numbers:
        # Determine all possible parent TreeNumbers
        parent_tn = ".".join(tn.split(".")[:-1]) if "." in tn else None

        if parent_tn and parent_tn in tree_to_ui:
            parent_descriptor_uis.add(tree_to_ui[parent_tn])

    # Store structured data (without TreeNumbers)
    mesh_hierarchy[descriptor_ui] = {
        "DescriptorName": descriptor_name,
        "ParentDescriptorUIs": list(parent_descriptor_uis), 
        "ScopeNote": scope_note
    }

for ui, details in mesh_hierarchy.items():
    if details["ScopeNote"]:
        details["ScopeNote"] = details["ScopeNote"].strip()

save_json(mesh_hierarchy, 'mesh.json')

drug_cat = open_json('drug_categories.json')

invalid_mesh_ids = []
for d in drug_cat.keys():
    if d not in mesh_hierarchy:
        invalid_mesh_ids.append(d)

drugs = open_json('drugs.json')

# Keep only valid drug categories
for drug, info in drugs.items():
    info['categories'] = [mesh for mesh in info['categories'] if mesh not in invalid_mesh_ids]

for mesh in invalid_mesh_ids:
    del drug_cat[mesh]

save_json(drugs, 'drugs.json')
save_json(drug_cat, 'drug_categories.json')