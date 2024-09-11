import os
import json


def load_metadata(folder: str):
    with open(os.path.join(folder, "metadata.json"), "r") as file:
        return json.load(file)


def save_metadata(folder: str, metadata: dict):
    # Ensure the folder exists
    os.makedirs(folder, exist_ok=True)

    # Define the path to the metadata file
    metadata_file = os.path.join(folder, "metadata.json")

    # Write the metadata dictionary to the JSON file
    with open(metadata_file, "w") as file:
        json.dump(metadata, file, indent=4)


def get_number_of_mappings(code_metadata):
    count = 0
    for value in code_metadata.values():
        count += len(value.get("parent_codes", []))
    return count
