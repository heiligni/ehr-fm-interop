import sys
import os

# Adding the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(os.path.join(parent_dir, "src"))


import argparse
import pickle
import pyarrow.csv
from datasets import load_from_disk
from femr.featurizers import join_labels
from femr.models.tokenizer import FEMRTokenizer
import torch
from datetime import datetime
import json


from utility.logs import log, log_time
from models.transformer import compute_features as compute_features_clmbr_t_base
from models.uni_transformer import compute_features as compute_features_uni_transformer
from models.clmbr_t_base import load_clmbr_t_base
from models.univeral_tokenizer import UniversalTokenizer


def select_best_model(base_dir):
    best_model_dir = None
    best_eval_loss = float("inf")  # Initialize with a very high value

    # Iterate through all subdirectories in the base directory
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)

        # Ensure it's a directory and contains a results.json file
        if os.path.isdir(subdir_path):
            results_file = os.path.join(subdir_path, "results.json")
            if os.path.exists(results_file):
                with open(results_file, "r") as f:
                    results = json.load(f)
                    eval_loss = results.get(
                        "eval_loss", float("inf")
                    )  # Default to infinity if eval_loss is missing

                    # Compare with the current best eval_loss
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        best_model_dir = subdir_path

    return best_model_dir, best_eval_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Features")
    parser.add_argument("--output-path", type=str)
    parser.add_argument("--num-procs", type=int, default=1)
    parser.add_argument("--model-name", type=str, required=True)

    args = parser.parse_args()

    num_procs = args.num_procs

    output_path = args.output_path
    log("GENERATING FEATURES")
    preprocessed_data_path = os.path.join(output_path, "preprocessed")

    before_dataset_loading = datetime.now()
    dataset = load_from_disk(preprocessed_data_path)
    after_dataset_loading = datetime.now()
    log_time(before_dataset_loading, after_dataset_loading, "load dataset")

    label_dir = os.path.join(output_path, "labels")
    csv_files = [file for file in os.listdir(label_dir) if file.endswith(".csv")]

    labels = []
    for file in csv_files:
        labels.extend(pyarrow.csv.read_csv(os.path.join(label_dir, file)).to_pylist())

    print(f"Total number of labels: {len(labels)}")
    assert len(labels) > 0, "No labels found. Please check the label directory."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {str(device)}")

    before_feature_computation = datetime.now()
    print(f"Starting featurization with {str(num_procs)} procs.")
    model_path = os.path.join(output_path, "fm")

    # Model does not exist if it was not pretrained before
    if args.model_name == "CLMBR-T-base":
        if not os.path.exists(model_path):
            print("Model does not exist. Loading pretrained model...")
            model, tokenizer, batch_processor = load_clmbr_t_base(None)
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
        else:
            print("Model already exists. Searching for the best model...")
            tokenizer = FEMRTokenizer.from_pretrained(model_path)

            # Select the best model from subdirectories
            best_model_dir, best_eval_loss = select_best_model(model_path)
            if best_model_dir:
                print(
                    f"Best model found in {best_model_dir} with eval_loss: {best_eval_loss}"
                )
                model_path = best_model_dir
    else:
        print("Lab model was selected. Searching for the best model...")
        tokenizer = UniversalTokenizer.from_pretrained(model_path)
        # Select the best model from subdirectories
        best_model_dir, best_eval_loss = select_best_model(model_path)
        if best_model_dir:
            print(
                f"Best lab model found in {best_model_dir} with eval_loss: {best_eval_loss}"
            )
            model_path = best_model_dir
        else:
            raise ValueError("No best model found. Please check the model directory.")
    if args.model_name == "CLMBR-T-base":
        features = compute_features_clmbr_t_base(
            dataset,
            model_path,
            tokenizer,
            labels,
            num_proc=num_procs,
            tokens_per_batch=2048,
            device=device,
            ontology=None,
        )
    elif args.model_name == "CLMBR-T-lab":
        features = compute_features_uni_transformer(
            dataset,
            model_path,
            tokenizer,
            labels,
            num_proc=num_procs,
            tokens_per_batch=2048,
            device=device,
        )
    after_feature_computation = datetime.now()
    log_time(before_feature_computation, after_feature_computation, "compute features")

    os.makedirs(os.path.join(output_path, "features"), exist_ok=True)
    feature_file_path = os.path.join(output_path, "features", "features.pkl")
    with open(feature_file_path, "wb") as feature_file:
        pickle.dump(features, feature_file)
    features_with_labels = {}
    for file in csv_files:
        labeler_name = file.split(".")[0]
        labels = pyarrow.csv.read_csv(os.path.join(label_dir, file)).to_pylist()
        features_with_labels[labeler_name] = join_labels(features, labels)
    file_path = os.path.join(output_path, "features", "features_and_labels.pkl")
    with open(file_path, "wb") as file:
        pickle.dump(features_with_labels, file)
