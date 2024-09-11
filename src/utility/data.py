from utility.logs import log_time
from datetime import datetime
from datasets import load_from_disk


def load_dataset(input_path):
    print(f"Loading dataset from {input_path}")
    start_time = datetime.now()
    dataset = load_from_disk(input_path)
    log_time(start_time, datetime.now(), "load dataset")
    return dataset


def save_dataset(dataset, output_path):
    print(f"Saving dataset to {output_path}")
    start_time = datetime.now()
    dataset.save_to_disk(output_path, num_proc=1)
    removed_files = dataset.cleanup_cache_files()
    print(f"Removed {removed_files} cache files")
    log_time(start_time, datetime.now(), "save dataset")
    result = dataset.cleanup_cache_files()
    print(f"Removed {str(result)} cache files")
