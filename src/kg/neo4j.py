from neo4j import GraphDatabase
from utils.utils import open_json, save_json

drugs = open_json('drugs_with_conditions_final.json')
conditions = open_json('conditions.json')
targets = open_json('targets.json')
proteins = open_json('proteins.json')
go = open_json('go.json')
atc = open_json('atc.json')
protein_map = open_json('protein_map.json')
go_name_to_id = open_json('go_name_to_id.json')
drug_categories = open_json('drug_categories.json')
mesh = open_json('mesh.json')

class Neo4jHandler:
    """
    A class to handle connections and queries for the Neo4j database.
    """
    def __init__(self, uri, user, password):
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            self.driver = None

    def close(self):
        """Closes the connection to the Neo4j database."""
        self.driver.close()

    def execute_query(self, query, parameters=None, write=True):
        """Executes a Cypher query in the Neo4j database."""
        try:
            with self.driver.session() as session:
                if write:
                    return session.execute_write(lambda tx: tx.run(query, parameters or {}).data())
                else:
                    return session.execute_read(lambda tx: tx.run(query, parameters or {}).data())
        except Exception as e:
            print(f"Query failed: {e}")
            return None

def create_atc_nodes(db, atc, group):
    """
    Creates ATC nodes with separate node types for each ATC group in the Neo4j database.
    """
    group_to_label = {
#         "anatomical": "AnatomicalGroup",
        "therapeutic": "TherapeuticGroup",
        "pharmacological": "PharmacologicalGroup",
        "chemical": "ChemicalGroup"
    }
    
    if group not in group_to_label:
        raise ValueError(f"Invalid group. Must be one of {list(group_to_label.keys())}")
    
    node_label = group_to_label[group]

    query = f"""
    UNWIND $atc AS atc_item
    MERGE (a:{node_label} {{atc_id: atc_item.atc_id}})
    SET a.name = atc_item.name
    """

    data = [
        {"atc_id": key, "name": value.get("name")}
        for key, value in atc.items()
        if len(key) > 1 and len(key) < 7 and value["group"] == group
    ]

    db.execute_query(query, {"atc": data})
                
def create_drug_nodes(db, drugs):
    """
    Creates Drug nodes in the Neo4j database.
    """
    query = """
    UNWIND $drugs AS drug
    MERGE (d:Drug {drugbank_id: drug.drugbank_id})
    SET d.name = drug.name
    """
    data = [
        {
            "drugbank_id": drug_id,
            "name": drug_data.get("name")
        }
        for drug_id, drug_data in drugs.items()
    ]

    if data:
        db.execute_query(query, {"drugs": data})
    
def create_drug_atc_edges(db, drugs):
    """
    Creates relationships between Drug and ATC nodes in the Neo4j database.
    """
    group_to_label = {
#         "anatomical_groups": ("AnatomicalGroup", "HAS_ANATOMICAL_GROUP"),
        "therapeutic_groups": ("TherapeuticGroup", "HAS_THERAPEUTIC_GROUP"),
        "pharmacological_groups": ("PharmacologicalGroup", "HAS_PHARMACOLOGICAL_GROUP"),
        "chemical_groups": ("ChemicalGroup", "HAS_CHEMICAL_GROUP"),
    }

    relationships = []

    for drugbank_id, data in drugs.items():
        for group, atc_ids in data.items():
            if group not in group_to_label:
                continue

            node_label, relationship_type = group_to_label[group]
            for atc_id in atc_ids:
                relationships.append({
                    "drugbank_id": drugbank_id,
                    "atc_id": atc_id,
                    "atc_label": node_label,
                    "relationship_type": relationship_type
                })

    query = """
    UNWIND $relationships AS rel
    MATCH (d:Drug {drugbank_id: rel.drugbank_id})
    MATCH (a:{node_label} {atc_id: rel.atc_id})
    MERGE (d)-[r:REL_TYPE]->(a)
    """

    for label, rel_type in group_to_label.values():
        db.execute_query(
            query.replace(":REL_TYPE", f":{rel_type}").replace("{node_label}", label),
            {"relationships": [rel for rel in relationships if rel["relationship_type"] == rel_type]},
        )

def create_drug_category_nodes(db, drug_categories):
    """
    Creates Drug Category nodes in the Neo4j database.
    """
    query = """
    UNWIND $drug_categories AS drug_category
    MERGE (d:CategoryDrug {category_id: drug_category.category_id})
    SET d.name = drug_category.name
    """

    data = [
        {"category_id": category_id, "name": name}
        for category_id, name in drug_categories.items()
    ]

    db.execute_query(query, {"drug_categories": data})
        
def create_drug_drug_category_edges(db, drugs):
    """
    Creates 'HAS_CATEGORY' relationships between Drug nodes and Drug Category nodes 
    in the Neo4j database.
    """
    query = """
    UNWIND $relationships AS rel
    MATCH (d:Drug {drugbank_id: rel.drugbank_id})
    MATCH (dc:CategoryDrug {category_id: rel.category_id})
    MERGE (d)-[:HAS_CATEGORY]->(dc)
    """
    
    relationships = []
    
    for drug_id, drug_data in drugs.items():
        categories = drug_data.get('categories', [])
        
        for category_id in categories:
            relationships.append({
                "drugbank_id": drug_id,
                "category_id": category_id
            })
    
    if relationships:
        db.execute_query(query, {"relationships": relationships})

def create_condition_nodes(db, conditions):
    """
    Creates Condition nodes in the Neo4j database.
    """
    query = """
    UNWIND $conditions AS condition
    MERGE (c:Condition {condition_id: condition.condition_id})
    SET c.name = condition.name
    """
    data = [
        {
            "condition_id": condition_id,
            "name": data.get("name")
        }
        for condition_id, data in conditions.items()
    ]
    db.execute_query(query, {"conditions": data})
    
def create_drug_condition_edges(db, drugs):
    """
    Creates 'INDICATED_FOR' relationships between Drug and Condition nodes in the Neo4j database.
    """
    query = """
    UNWIND $relationships AS rel
    MATCH (c:Condition {condition_id: rel.condition_id})
    MATCH (d:Drug {drugbank_id: rel.drugbank_id})
    MERGE (d)-[:INDICATED_FOR]->(c)
    """
    
    relationships = [
        {"condition_id": condition_id, "drugbank_id": drugbank_id}
        for drugbank_id, data in drugs.items()
        for condition_id in data.get("treats", [])
        if condition_id and drugbank_id 
    ]
    
    if relationships: 
        db.execute_query(query, {"relationships": relationships})
        
def create_go_nodes(db, go, group):
    """
    Creates GO nodes with separate node types for each GO group in the Neo4j database.
    """
    group_to_label = {
        "biological_process": "BiologicalProcess",
        "molecular_function": "MolecularFunction",
        "cellular_component": "CellularComponent",
    }
    
    if group not in group_to_label:
        raise ValueError(f"Invalid group. Must be one of {list(group_to_label.keys())}")
    
    node_label = group_to_label[group]

    query = f"""
    UNWIND $go AS go_item
    MERGE (g:{node_label} {{go_id: go_item.go_id}})
    SET g.name = go_item.name
    """

    data = [
        {"go_id": key, "name": value.get("name")}
        for key, value in go.items()
        if value["namespace"] == group
    ]

    db.execute_query(query, {"go": data})
        
def create_protein_nodes(db, proteins):
    """
    Creates Protein nodes in the Neo4j database.
    """
    query = """
    UNWIND $proteins AS protein
    MERGE (p:Protein {uniprot_id: protein.uniprot_id})
    SET p.name = protein.name
    """
    
    data = [
        {"uniprot_id": uniprot_id, "name": data.get("name")}
        for uniprot_id, data in proteins.items()
        if uniprot_id and data.get("name") 
    ]
    
    db.execute_query(query, {"proteins": data})

def create_drug_protein_edges(db, targets, protein_map):
    """
    Creates 'TARGETS' relationships between Drug and Protein nodes in the Neo4j database.
    """
    query = """
    UNWIND $relationships AS rel
    MATCH (d:Drug {drugbank_id: rel.drugbank_id})
    MATCH (p:Protein {uniprot_id: rel.uniprot_id})
    MERGE (d)-[r:TARGETS]->(p)
    SET r.actions = rel.actions
    """
    
    relationships = []
    for drug_id, proteins in targets.items():
        for protein_data in proteins:
            protein_name = protein_data.get('name')
            uniprot_id = protein_map.get(protein_name)
            actions = protein_data.get('actions')
            known_action = protein_data.get('known_action')

            if uniprot_id and actions and known_action:
                if known_action.lower() == 'yes':
                    if actions[0] in ('other', 'unknown', 'other/unknown'):
                        actions = ['unknown']

                    actions_str = '|'.join(actions)

                    relationships.append({
                        "drugbank_id": drug_id,
                        "uniprot_id": uniprot_id,
                        "actions": actions_str,
                    })

    if relationships:
        db.execute_query(query, {"relationships": relationships})

def create_condition_protein_edges(db, proteins):
    """
    Creates 'ASSOCIATED_WITH' relationships between Protein and Condition nodes in the Neo4j database.
    """
    query = """
    UNWIND $relationships AS rel
    MATCH (c:Condition {condition_id: rel.condition_id})
    MATCH (p:Protein {uniprot_id: rel.uniprot_id})
    MERGE (p)-[r:ASSOCIATED_WITH]->(c)
    """

    relationships = [
    {
        "condition_id": condition,
        "uniprot_id": uniprot_id,
    }
    for uniprot_id, data in proteins.items()
    for condition in data.get("conditions", [])
    ]
    
    if relationships:
        db.execute_query(query, {"relationships": relationships})


def create_protein_go_edges(db, proteins, go_name_to_id):
    """
    Creates relationships between Protein and GO nodes in the Neo4j database.
    """
    group_to_label = {
        "process": ("BiologicalProcess", "HAS_PROCESS"),
        "function": ("MolecularFunction", "HAS_FUNCTION"),
        "component": ("CellularComponent", "HAS_COMPONENT"),
    }

    relationships = []

    for uniprot_id, data in proteins.items():
        for group, go_names in data["go_classifiers"].items():
            node_label, relationship_type = group_to_label[group]
            for go_name in go_names:
                go_id = go_name_to_id.get(go_name)
                if go_id:
                    relationships.append({
                        "uniprot_id": uniprot_id,
                        "go_id": go_id,
                        "go_label": node_label,
                        "relationship_type": relationship_type
                    })

    query = """
    UNWIND $relationships AS rel
    MATCH (p:Protein {uniprot_id: rel.uniprot_id})
    MATCH (g:{node_label} {go_id: rel.go_id})
    MERGE (p)-[r:REL_TYPE]->(g)
    """

    for label, rel_type in group_to_label.values():
        db.execute_query(
            query.replace(":REL_TYPE", f":{rel_type}").replace("{node_label}", label),
            {"relationships": [rel for rel in relationships if rel["relationship_type"] == rel_type]},
        )

def delete_isolated_nodes(db):
    """
    Deletes all isolated nodes from the Neo4j database.
    """
    query = """
    MATCH (n)
    WHERE NOT (n)--()
    DELETE n
    """
    db.execute_query(query)

def main1():
    """
    Main function to connect to the database and process the dictionaries.
    """
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "password"

    db = Neo4jHandler(uri, user, password)

    try:
        for group in ['therapeutic', 'pharmacological', 'chemical']:
            create_atc_nodes(db, atc, group)
        create_drug_nodes(db, drugs)
        create_drug_atc_edges(db, drugs)
        create_drug_category_nodes(db, drug_categories)
        create_drug_drug_category_edges(db, drugs)
        create_condition_nodes(db, conditions)
        create_drug_condition_edges(db, drugs)
        for group in ['biological_process', 'molecular_function', 'cellular_component']:
            create_go_nodes(db, go, group)
        create_protein_nodes(db, proteins)
        create_drug_protein_edges(db, targets, protein_map)
        create_condition_protein_edges(db, proteins)
        create_protein_go_edges(db, proteins, go_name_to_id)
        delete_isolated_nodes(db)
        
    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        db.close()

if __name__ == "__main__":
    main1()

class Neo4jLoader:
    """
    A class to load data from a Neo4j database.
    """

    def __init__(self, uri, user, password):
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            print("Connected to Neo4j successfully.")
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            self.driver = None
            
    def close(self):
        """Closes the connection to the Neo4j database."""
        if self.driver:
            self.driver.close()

    def load_query(self, query, parameters=None):
        """Executes a read query and returns the results."""
        try:
            with self.driver.session() as session:
                results = session.execute_read(lambda tx: tx.run(query, parameters or {}).data())
                if not results:
                    print(f"Warning: Query returned no results.\nQuery: {query}")
                return results
        except Exception as e:
            print(f"Query failed: {e}")
            return None

    def load_anatomical_groups(self):
        """Loads all anatomical ATC code nodes from Neo4j."""
        query = """
        MATCH (a:AnatomicalGroup)
        RETURN a.atc_id AS atc_id, a.name AS name
        """
        return self.load_query(query)
    
    def load_pharmacological_groups(self):
        """Loads all pharmacological ATC code nodes from Neo4j."""
        query = """
        MATCH (a:PharmacologicalGroup)
        RETURN a.atc_id AS atc_id, a.name AS name
        """
        return self.load_query(query)
    
    def load_therapeutic_groups(self):
        """Loads all therapeutic ATC code nodes from Neo4j."""
        query = """
        MATCH (a:TherapeuticGroup)
        RETURN a.atc_id AS atc_id, a.name AS name
        """
        return self.load_query(query)
    
    def load_chemical_groups(self):
        """Loads all chemical ATC code nodes from Neo4j."""
        query = """
        MATCH (a:ChemicalGroup)
        RETURN a.atc_id AS atc_id, a.name AS name
        """
        return self.load_query(query)

    def load_drugs(self):
        """Loads all drug nodes from Neo4j."""
        query = """
        MATCH (d:Drug)
        RETURN d.drugbank_id AS drugbank_id, d.name AS name
        """
        return self.load_query(query)

    def load_conditions(self):
        """Loads all condition nodes from Neo4j."""
        query = """
        MATCH (c:Condition)
        RETURN c.condition_id AS condition_id, c.name AS name
        """
        return self.load_query(query)
    
    def load_biological_processes(self):
        "Loads all biological process nodes from Neo4j."
        query = """
        MATCH (g:BiologicalProcess)
        RETURN g.go_id AS go_id, g.name AS name"""
        
        return self.load_query(query)
        
    def load_molecular_functions(self):
        "Loads all molecular function nodes from Neo4j."
        query = """
        MATCH (g:MolecularFunction)
        RETURN g.go_id AS go_id, g.name AS name"""
        
        return self.load_query(query)

    def load_cellular_components(self):
        "Loads all cellular component nodes from Neo4j."
        query = """
        MATCH (g:CellularComponent)
        RETURN g.go_id AS go_id, g.name AS name"""
        
        return self.load_query(query)

    def load_drug_categories(self):
        "Loads all drug category nodes from Neo4j."
        query = """
        MATCH (d:CategoryDrug)
        RETURN d.category_id AS category_id, d.name AS name"""

        return self.load_query(query)
    
    def load_drug_categories_drug_edges(self):
        "Loads drug-drug categories relationships."
        query = """
        MATCH (d:Drug)-[:HAS_CATEGORY]->(d1:CategoryDrug)
        RETURN d.drugbank_id AS drugbank_id, d1.category_id AS category_id"""
        
        return self.load_query(query)

    def load_proteins(self):
        """Loads all protein nodes from Neo4j."""
        query = """
        MATCH (p:Protein)
        RETURN p.uniprot_id AS uniprot_id, p.name AS name
        """
        return self.load_query(query)
    
    def load_drug_anatomical_edges(self):
        """Loads drug-anatomical ATC relationships."""
        query = """
        MATCH (d:Drug)-[:HAS_ANATOMICAL_GROUP]->(a:AnatomicalGroup)
        RETURN d.drugbank_id AS drugbank_id, a.atc_id AS atc_id
        """
        return self.load_query(query)
    
    def load_drug_therapeutic_edges(self):
        """Loads drug-therapeutic ATC relationships."""
        query = """
        MATCH (d:Drug)-[:HAS_THERAPEUTIC_GROUP]->(a:TherapeuticGroup)
        RETURN d.drugbank_id AS drugbank_id, a.atc_id AS atc_id
        """
        return self.load_query(query)
    
    def load_drug_pharmacological_edges(self):
        """Loads drug-pharmacological ATC relationships."""
        query = """
        MATCH (d:Drug)-[:HAS_PHARMACOLOGICAL_GROUP]->(a:PharmacologicalGroup)
        RETURN d.drugbank_id AS drugbank_id, a.atc_id AS atc_id
        """
        return self.load_query(query)
    
    def load_drug_chemical_edges(self):
        """Loads drug-chemical ATC relationships."""
        query = """
        MATCH (d:Drug)-[:HAS_CHEMICAL_GROUP]->(a:ChemicalGroup)
        RETURN d.drugbank_id AS drugbank_id, a.atc_id AS atc_id
        """
        return self.load_query(query)
    
    def load_drug_condition_edges(self):
        """Loads drug-condition indications."""
        query = """
        MATCH (d:Drug)-[:INDICATED_FOR]->(c:Condition)
        RETURN d.drugbank_id AS drugbank_id, c.condition_id AS condition_id
        """
        return self.load_query(query)

    def load_drug_protein_edges(self):
        """Loads drug-protein relationships."""
        query = """
        MATCH (d:Drug)-[r:TARGETS]->(p:Protein)
        RETURN d.drugbank_id AS drugbank_id, p.uniprot_id AS uniprot_id, r.actions AS actions
        """
        return self.load_query(query)
    
    def load_condition_protein_edges(self):
        """Loads drug-protein relationships."""
        query = """
        MATCH (p:Protein)-[:ASSOCIATED_WITH]->(c:Condition)
        RETURN p.uniprot_id AS uniprot_id, c.condition_id AS condition_id
        """
        return self.load_query(query)
    
    def load_go_component_edges(self):
        """Loads GO-component relationships."""
        query = """
        MATCH (p:Protein)-[:HAS_COMPONENT]->(g:CellularComponent)
        RETURN p.uniprot_id AS uniprot_id, g.go_id AS go_id
        """
        return self.load_query(query)
    
    def load_go_function_edges(self):
        """Loads GO-function relationships."""
        query = """
        MATCH (p:Protein)-[:HAS_FUNCTION]->(g:MolecularFunction)
        RETURN p.uniprot_id AS uniprot_id, g.go_id AS go_id
        """
        return self.load_query(query)
    
    def load_go_process_edges(self):
        """Loads GO-process relationships."""
        query = """
        MATCH (p:Protein)-[:HAS_PROCESS]->(g:BiologicalProcess)
        RETURN p.uniprot_id AS uniprot_id, g.go_id AS go_id
        """
        return self.load_query(query)

class GraphExtractor:
    """
    Converts Neo4j graph data into triples (head, relation, tail) and extracts node IDs by type.
    """

    def __init__(self, neo4j_loader):
        self.loader = neo4j_loader

    def extract_triples(self):
        triples = []
        relation_queries = {
            "HAS_THERAPEUTIC_GROUP": self.loader.load_drug_therapeutic_edges,
            "HAS_PHARMACOLOGICAL_GROUP": self.loader.load_drug_pharmacological_edges,
            "HAS_CHEMICAL_GROUP": self.loader.load_drug_chemical_edges,
            "INDICATED_FOR": self.loader.load_drug_condition_edges,
            "TARGETS": self.loader.load_drug_protein_edges,
            "ASSOCIATED_WITH": self.loader.load_condition_protein_edges,
            "HAS_COMPONENT": self.loader.load_go_component_edges,
            "HAS_FUNCTION": self.loader.load_go_function_edges,
            "HAS_PROCESS": self.loader.load_go_process_edges,
            "HAS_CATEGORY": self.loader.load_drug_categories_drug_edges
        }

        for relation, query_func in relation_queries.items():
            results = query_func() or []
            
            for record in results:
                keys = list(record.keys())
                head, tail = record[keys[0]], record[keys[1]]
                attributes = {k: v for k, v in record.items() if k not in [keys[0], keys[1]]}

                triples.append((head, relation, tail))

        return triples

    def extract_nodes(self):
        """
        Extracts unique node IDs grouped by node type.

        Returns:
            dict: A dictionary where keys are node types and values are lists of node IDs.
        """
        node_queries = {
#             "AnatomicalGroup": self.loader.load_anatomical_groups,
            "PharmacologicalGroup": self.loader.load_pharmacological_groups,
            "TherapeuticGroup": self.loader.load_therapeutic_groups,
            "ChemicalGroup": self.loader.load_chemical_groups,
            "Drug": self.loader.load_drugs,
            "Condition": self.loader.load_conditions,
            "BiologicalProcess": self.loader.load_biological_processes,
            "MolecularFunction": self.loader.load_molecular_functions,
            "CellularComponent": self.loader.load_cellular_components,
            "CategoryDrug": self.loader.load_drug_categories,
            "Protein": self.loader.load_proteins,
        }

        node_dict = {}

        for node_type, query_func in node_queries.items():
            results = query_func() or []
            
            if results:
                key = list(results[0].keys())[0]
                name = list(results[0].keys())[1]
                node_dict[node_type] = [(record[key], record[name]) for record in results]
            else:
                node_dict[node_type] = []

        return node_dict
    
def main2():
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "password"

    try:
        loader = Neo4jLoader(uri, user, password)
        if loader.driver is None:
            print("Failed to connect to Neo4j. Exiting...")
            return  # Exit if the connection is not established
        
        extractor = GraphExtractor(loader)
        graph_triples = extractor.extract_triples()
        print(f"Extracted {len(graph_triples)} triples.")
        save_json(graph_triples, 'graph_triples.json')
        print("Triples saved to graph_triples.json.")
        graph_nodes = extractor.extract_nodes()
        save_json(graph_nodes, 'graph_nodes.json')

    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        if loader:
            loader.close()

if __name__ == "__main__":
    main2()

graph_triples = open_json('graph_triples.json')
graph_nodes = open_json('graph_nodes.json')

with open('graph_triples.txt', 'w') as f:
    for triple in graph_triples:
        f.write(" ".join(triple) + "\n")
        