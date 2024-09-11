"""We separate the creation of the tokenizer and pretraining batches from the training because this allows us to decrease the time on GPU nodes"""

import sys
import os

# Adding the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(os.path.join(parent_dir, "src"))

import argparse
from femr.index import PatientIndex
from femr.models.tasks import CLMBRTask
from femr.models.tokenizer import train_tokenizer as train_clmbr_tokenizer
from datetime import datetime

from models.processor import FEMRBatchProcessor
from split import PatientSplit
from utility.data import load_dataset, save_dataset
from transforms.mimic_transforms import lab_measurement_evaluation_batched
from models.universal_processor import UniFEMRBatchProcessor
from models.univeral_tokenizer import train_tokenizer as train_lab_tokenizer
from models.model_names import MODEL_NAMES
from utility.logs import log_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pretraining batches")
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--output-path", type=str)
    parser.add_argument("--num-procs", type=int, default=1)
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--model-name", type=str, choices=MODEL_NAMES)

    args = parser.parse_args()
    output_path = args.output_path

    ds = load_dataset(args.data_dir)

    model_path = os.path.join(output_path, "fm")
    os.makedirs(model_path, exist_ok=True)

    if args.model_name == "CLMBR-T-lab":
        print(
            "Preparing measurements for adjusted tokenization. Processing reference ranges..."
        )
        before_processing = datetime.now()
        ds = ds.map(
            lambda batch: lab_measurement_evaluation_batched(batch),
            batched=True,
            batch_size=100,
            num_proc=args.num_procs,
        )
        log_time(before_processing, datetime.now(), "Processing reference ranges")
        save_dataset(ds, os.path.join(output_path, "lab_tokenization_data"))

    split = PatientSplit.load_from_csv(os.path.join(output_path, "splits/split.csv"))
    split_ds = split.split_dataset(ds, PatientIndex(ds, num_proc=args.num_procs))
    split_ds.pop("test")
    if args.model_name == "CLMBR-T-lab":
        print("Train CLMBR-T-lab tokenizer")
        tokenizer = train_lab_tokenizer(
            split_ds["train"],
            vocab_size=args.vocab_size,
            num_proc=int(args.num_procs / 4),
        )
    else:
        print("Train default tokenizer")
        tokenizer = train_clmbr_tokenizer(
            split_ds["train"],
            vocab_size=args.vocab_size,
            num_proc=int(args.num_procs / 4),
        )

    tokenizer.save_pretrained(model_path)

    clmbr_task = CLMBRTask(clmbr_vocab_size=8192)
    if args.model_name == "CLMBR-T-lab":
        processor = UniFEMRBatchProcessor(tokenizer, clmbr_task)
    else:
        processor = FEMRBatchProcessor(tokenizer, clmbr_task)

    before_processing = datetime.now()
    train_batches = processor.convert_dataset(
        split_ds,
        tokens_per_batch=2048,
        min_patients_per_batch=1,
        num_proc=int(args.num_procs / 2),
    )
    log_time(before_processing, datetime.now(), "Converting dataset")
    print(f"Final dataset sizes: {train_batches}")

    save_dataset(train_batches, os.path.join(output_path, "train_batches"))
