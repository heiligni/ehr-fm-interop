import os
import pickle
import numpy as np
from meds import Patient


def get_features(output_dir_path: str):
    feature_file_path = os.path.join(output_dir_path, "features", "features.pkl")

    # Open the file in read binary mode and load the features
    with open(feature_file_path, "rb") as feature_file:
        return pickle.load(feature_file)


def get_features_for_patient_id(output_dir_path: str, patient_id: int):
    features = get_features(output_dir_path)
    patient_filter = np.where(features["patient_ids"] == patient_id)
    return {
        "patient_ids": features["patient_ids"][patient_filter],
        "feature_times": features["feature_times"][patient_filter],
        "features": features["features"][patient_filter],
    }


def get_patient_idx_ds(ds, patient_id):
    for idx, running_pat_id in enumerate(ds["patient_id"]):
        if running_pat_id == patient_id:
            return idx
    return None


def get_ontology(output_dir_path: str):
    ontology_file_path = os.path.join(output_dir_path, "ontology.pkl")
    with open(ontology_file_path, "rb") as f:
        return pickle.load(f)


def has_code(patient: Patient, code):
    for event in patient["events"]:
        for measurement in event["measurements"]:
            if measurement["code"] == code:
                return True
    return False
