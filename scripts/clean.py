import sys
import os

# Adding the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(os.path.join(parent_dir, "src"))


import argparse
from datasets import Dataset
from datetime import datetime

from utility.logs import log, log_time
from cleaning.cleaning import clean_patients

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean Data")
    parser.add_argument("--input-path", type=str)
    parser.add_argument("--output-path", type=str)
    parser.add_argument("--num-procs", type=int, default=1)
    parser.add_argument("--reduced-size", type=int, required=False)

    args = parser.parse_args()
    num_proc = args.num_procs

    log("Clean data")

    print(f"Load dataset using {num_proc} procs")
    before_dataset_loading = datetime.now() 
    dataset = Dataset.from_parquet(os.path.join(args.input_path, "data/*"), num_proc=num_proc)
    after_dataset_loading = datetime.now()
    log_time(before_dataset_loading, after_dataset_loading, "load dataset")
    
    if args.reduced_size is not None:
        print(f"Reducing dataset to size {args.reduced_size}")
        dataset = dataset.shuffle().select(range(0, args.reduced_size))
        
    
    cleaned_patients = clean_patients(dataset, args.num_procs)
    after_cleaning = datetime.now()
    log_time(after_dataset_loading, after_cleaning, "clean data")
    
    cleaned_path = os.path.join(args.output_path, "clean")
    if not os.path.exists(cleaned_path):
        os.makedirs(cleaned_path)

    print(f"Saving dataset to disk with {num_proc} procs")
    cleaned_patients.save_to_disk(cleaned_path, num_proc=num_proc)
    print(f"Saved dataset to {cleaned_path}")
    
    print("Cleanup cache files")
    result = dataset.cleanup_cache_files()
    result += cleaned_patients.cleanup_cache_files()
    print(f"Removed {str(result)} cache files")
