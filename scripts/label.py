import sys
import os

# Adding the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(os.path.join(parent_dir, "src"))

from typing import List
import argparse
import pyarrow
import pyarrow.csv
from meds import label


from utility.logs import log
from labelers.mortality import MortalityLabeler
from labelers.long_los import LongLOSLabeler
from labelers.disease_labeler import DiseaseLabeler
from utility.codes import HYPERLIPIDAEMIA_CODES, CKD_CODES, AKI_CODES
from utility.data import load_dataset


LABELERS: List[str] = [
    "mortality",
    "long_los",
    "hyperlipidemia_eos",
    "hyperlipidemia_ny",
    "ckd_eos",
    "ckd_ny",
    "aki_eos",
    "aki_ny",
]


def initialize_labeler(
    labeler_name: str, random_admission: bool, biased_admission: bool
):
    if labeler_name == "mortality":
        return MortalityLabeler(random_admission, biased_admission)
    elif labeler_name == "long_los":
        return LongLOSLabeler(random_admission, biased_admission)
    elif labeler_name == "hyperlipidemia_eos":
        return DiseaseLabeler(HYPERLIPIDAEMIA_CODES, "hyperlipidemia", "end_of_stay")
    elif labeler_name == "hyperlipidemia_ny":
        return DiseaseLabeler(HYPERLIPIDAEMIA_CODES, "hyperlipidemia", "next_year")
    elif labeler_name == "ckd_eos":
        return DiseaseLabeler(CKD_CODES, "ckd", "end_of_stay")
    elif labeler_name == "ckd_ny":
        return DiseaseLabeler(CKD_CODES, "ckd", "next_year")
    elif labeler_name == "aki_eos":
        return DiseaseLabeler(AKI_CODES, "aki", "end_of_stay")
    elif labeler_name == "aki_ny":
        return DiseaseLabeler(AKI_CODES, "aki", "next_year")
    else:
        raise ValueError(f"Unsupported labeler: {labeler_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run femr labeler")
    parser.add_argument("--input-path", type=str)
    parser.add_argument("--output-path", type=str)
    parser.add_argument(
        "--labeler",
        type=str,
        help="Name of labeling function to create.",
        choices=LABELERS,
    )
    parser.add_argument("--biased-admission-selection", action="store_true")
    parser.add_argument("--random-admission-selection", action="store_true")

    args = parser.parse_args()

    print(f"Creating {args.labeler} labeler")
    print(f"Random admission: {args.random_admission_selection}")
    print(f"Biased admission selection: {args.biased_admission_selection}")
    labeler = initialize_labeler(
        args.labeler, args.random_admission_selection, args.biased_admission_selection
    )
    print(f"Labeler {args.labeler} initialized")

    dataset = load_dataset(args.input_path)

    log("LABELLING")
    labeler_name = str(labeler)
    print(f"Generating labels with labeler: {labeler_name}")
    labeled_patients = labeler.apply(dataset, 1)
    table = pyarrow.Table.from_pylist(labeled_patients, schema=label)

    label_dir = os.path.join(args.output_path, "labels")
    os.makedirs(label_dir, exist_ok=True)
    pyarrow.csv.write_csv(table, os.path.join(label_dir, labeler_name + ".csv"))
