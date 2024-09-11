import sys
import os


# Adding the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

import argparse
from datasets import load_from_disk
from femr.hf_utils import aggregate_over_dataset
from src.ehr_stats.codes import create_code_occurence_plot, create_code_stats_table, plot_codes_per_patient, extract_code_stats, combine_code_stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate code stats")
    parser.add_argument("--output-path", type=str)
    parser.add_argument("--num-procs", type=int, default=1)

    args = parser.parse_args()

    clean_dataset = load_from_disk(os.path.join(args.output_path, "clean"))
    preprocessed_dataset = load_from_disk(os.path.join(args.output_path, "preprocessed"))
    code_stats_raw = aggregate_over_dataset(clean_dataset, extract_code_stats, combine_code_stats, 500, args.num_procs)
    code_stats_preprocessed = aggregate_over_dataset(preprocessed_dataset, extract_code_stats, combine_code_stats, 500, args.num_procs)

    plot_dir = os.path.join(args.output_path, "plots")
    table_dir = os.path.join(args.output_path, "tables")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)

    create_code_occurence_plot(code_stats_preprocessed, plot_dir)
    create_code_stats_table(code_stats_raw, code_stats_preprocessed, table_dir)
    plot_codes_per_patient(code_stats_preprocessed['unique_code_count'], plot_dir)