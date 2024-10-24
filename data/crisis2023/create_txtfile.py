import json

# Load the JSON data from node_to_id.json
with open('node_to_id.json', 'r') as json_file:
    node_to_id = json.load(json_file)

# Write the entity2id.txt file
with open('entity2id.txt', 'w') as entity_file:
    for entity, entity_id in node_to_id.items():
        entity_file.write(f"{entity}\t{entity_id}\n")

print("entity2id.txt has been created successfully.")