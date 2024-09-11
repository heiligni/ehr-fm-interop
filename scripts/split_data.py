import os
import sys

# Adding the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(os.path.join(parent_dir, "src"))


from split import create_and_save_split
from utility.logs import log

import argparse
from datasets import load_from_disk

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate splits")
    parser.add_argument("--input-path", type=str)
    parser.add_argument("--output-path", type=str)
    parser.add_argument("--test-frac", type=float)
    parser.add_argument("--val-frac", type=float)


    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    TEST_FRAC = args.test_frac
    VAL_FRAC = args.val_frac
    TRAIN_FRAC = 1 - TEST_FRAC - VAL_FRAC

    split_dir_path = os.path.join(output_path, "splits")
    os.makedirs(split_dir_path, exist_ok=True)
    split_file_path = os.path.join(split_dir_path, "split.csv")

    log("GENERATING SPLITS")
    dataset = load_from_disk(input_path)
    create_and_save_split(dataset, TEST_FRAC, VAL_FRAC, TRAIN_FRAC, split_file_path)