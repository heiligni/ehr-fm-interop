import os
import sys

# Adding the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(os.path.join(parent_dir, "src"))

SEED = 12345

import argparse
from datetime import datetime

from utility.filter import adm_filter_function, adm_type_filter
from utility.logs import log
from utility.data import load_dataset, save_dataset


def filter_dataset(dataset, filter_function, num_proc):
    print(f"Filtering dataset with {filter_function} using {num_proc} processes")
    start_time = datetime.now()

    if filter_function == "admission_length":
        dataset = dataset.filter(
            adm_filter_function, load_from_cache_file=False, num_proc=num_proc
        )
    elif filter_function == "admission_type":
        dataset = dataset.filter(
            adm_type_filter, load_from_cache_file=False, num_proc=num_proc
        )

    end_time = datetime.now()
    print(f"Dataset filtered in {end_time - start_time}")
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subsample Dataset")
    parser.add_argument("--input-path", type=str)
    parser.add_argument("--output-path", type=str)
    parser.add_argument("--num-procs", type=int, default=1)
    parser.add_argument("--filter-function", type=str)
    parser.add_argument("--size", type=int, required=False)

    args = parser.parse_args()

    output_path = args.output_path
    input_path = args.input_path
    num_proc = args.num_procs
    filter_function = args.filter_function

    log("Cohort Selection")
    print(f"Num procs: {num_proc}")

    os.makedirs(output_path, exist_ok=True)

    dataset = load_dataset(input_path)
    print(f"Dataset loaded with {len(dataset)} records")

    after_loading_data = datetime.now()

    dataset = filter_dataset(dataset, filter_function, num_proc)
    print(f"Dataset filtered to {len(dataset)} records")

    if args.size is not None:
        size = args.size
        print(f"Selecting random cohort of size {size}")
        assert (
            len(dataset) >= size
        ), "The dataset size is smaller than the requested size."
        dataset = dataset.shuffle(SEED).select(range(0, size))
    save_dataset(dataset, output_path)
