import sys
import os

# Adding the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(os.path.join(parent_dir, "src"))


import argparse
import pickle
import json
from femr.ontology import Ontology
from datetime import datetime

from utility.logs import log, log_time
from utility.data import load_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-path", type=str)
    parser.add_argument("--metadata-path", type=str)
    parser.add_argument("--input-path", type=str)
    parser.add_argument("--num-procs", type=int, default=1)
    parser.add_argument("--output-path", type=str)

    args = parser.parse_args()

    num_proc = args.num_procs

    log("Process Ontology")

    ontology_file_path = os.path.join(args.output_path, "ontology.pkl")

    if os.path.exists(ontology_file_path):
        print(
            "Ontology was already processed. Skipping ontology processing. Loading ontology from file"
        )
    else:
        with open(args.metadata_path, "r") as file:
            data_metadata = json.load(file)
            print("Loading Ontology")
            ontology = Ontology(args.vocab_path, data_metadata["code_metadata"])
            dataset = load_dataset(args.input_path)

            print("Start pruning ontology")
            before_pruning = datetime.now()
            ontology.prune_to_dataset(dataset, num_proc=4)
            after_pruning = datetime.now()
            log_time(before_pruning, after_pruning, "pruning ontology")

            dataset.cleanup_cache_files()
            os.makedirs(args.output_path, exist_ok=True)
            print(f"Storing ontology to {ontology_file_path}")
            with open(ontology_file_path, "wb") as f:
                pickle.dump(ontology, f)
