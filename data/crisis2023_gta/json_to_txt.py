import json

# Read the JSON file
with open('./node_to_id.json', 'r') as json_file:
    data = json.load(json_file)

# Create a text file and write the key-value pairs with tabs
with open('./entity2id.txt', 'w') as txt_file:
    for key, value in data.items():
        txt_file.write(f"{key}\t{value}\n")
