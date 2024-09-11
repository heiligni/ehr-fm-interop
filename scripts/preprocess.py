import sys
import os


# Adding the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(os.path.join(parent_dir, "src"))

import pickle
import argparse
from datasets import load_from_disk
from meds import death_code, birth_code
from datetime import datetime
from femr.models.tokenizer import FEMRTokenizer

from models.clmbr_t_base import get_tokenizer
from utility.logs import log, log_time
from transforms.remove_codes_not_in_clmbr import convert_patient_to_clmbr
from models.model_names import MODEL_NAMES, CLMBR_T_BASE, CLMBR_T_LAB
from models.univeral_tokenizer import UniversalTokenizer

"""
Keeping the VISIT_CODE helps because multiple same codes at the same day are filtered out when preprocessing.
We need visit information to generate our labels.
"""
keep_codes = [death_code, birth_code]


def load_tokenizer(output_path, model_name):
    model_dir = os.path.join(output_path, "fm")
    if model_name == CLMBR_T_BASE:
        if os.path.exists(model_dir):
            print(f"Loading custom trained tokenizer from {model_dir}")
            tokenizer = FEMRTokenizer.from_pretrained(model_dir)
        else:
            print("Loading default tokenizer")
            tokenizer = get_tokenizer(None)
    elif model_name == CLMBR_T_LAB:
        print(f"Loading custom lab tokenizer from {model_dir}")
        tokenizer = UniversalTokenizer.from_pretrained(model_dir)
    return tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate code stats")
    parser.add_argument("--output-path", type=str)
    parser.add_argument("--input-path", type=str)
    parser.add_argument("--num-procs", type=int, default=1)
    parser.add_argument("--model-name", type=str, choices=MODEL_NAMES)

    args = parser.parse_args()
    num_proc = args.num_procs

    log("PREPROCESSING")
    dataset = load_from_disk(args.input_path)
    tokenizer = load_tokenizer(args.output_path, args.model_name)
    ontology_file_path = os.path.join(args.output_path, "ontology.pkl")
    with open(ontology_file_path, "rb") as f:
        ontology = pickle.load(f)

    print(f"Starting preprocessing with {str(args.num_procs)} procs.")
    before_preprocessing = datetime.now()
    preprocessed_dataset = dataset.map(
        lambda patient: convert_patient_to_clmbr(
            patient, tokenizer, ontology, keep_codes
        ),
        num_proc=num_proc,
        load_from_cache_file=False,
    )
    after_preprocessing = datetime.now()
    log_time(before_preprocessing, after_preprocessing, "preprocessing")

    dataset.cleanup_cache_files()

    # Save data
    preprocessed_path = os.path.join(args.output_path, "preprocessed")
    os.makedirs(preprocessed_path, exist_ok=True)
    preprocessed_dataset.save_to_disk(preprocessed_path, num_proc=1)

    preprocessed_dataset.cleanup_cache_files()
