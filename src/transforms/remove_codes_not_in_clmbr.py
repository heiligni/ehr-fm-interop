from meds import Patient, Event
from femr.models.tokenizer import FEMRTokenizer
from femr.ontology import Ontology
from copy import deepcopy

from utility.codes import (
    VISIT_CODE,
    ADMISSION_PREFIX,
    CKD_CODES,
    HYPERLIPIDAEMIA_CODES,
    AKI_CODES,
)
from transforms.mimic_transforms import reorder_measurements

DISEASE_LABEL_CODES = set()
DISEASE_LABEL_CODES |= CKD_CODES
DISEASE_LABEL_CODES |= HYPERLIPIDAEMIA_CODES
DISEASE_LABEL_CODES |= AKI_CODES


def convert_patient_to_clmbr(
    patient: Patient, tokenizer: FEMRTokenizer, ontology: Ontology, keep_codes=[]
) -> Patient:
    converted_patient: Patient = {
        "patient_id": patient["patient_id"],
        "static_measurements": [],
        "events": [],
    }

    current_date = None

    for event in patient["events"]:
        # This is required as the model only has a resolution of seconds and otherwise the labels don't match the timestamps exactly
        event["time"] = event["time"].replace(microsecond=0)

        if event["time"].date() != current_date:
            current_date = event["time"].date()
            codes_seen_today = set()

        new_measurements = []
        added_admission_id = None
        for measurement in event["measurements"]:
            if measurement["code"].startswith(ADMISSION_PREFIX):
                admission_type = measurement["code"].split("/")[1]
                measurement["code"] = VISIT_CODE
                measurement["metadata"]["priority"] = admission_type
                added_admission_id = measurement["metadata"]["visit_id"]

            features, _ = tokenizer.get_feature_codes(event["time"], measurement)

            # If we train the tokenizer ourselves, the visit code must not be in the tokenizer
            if measurement["code"] in keep_codes or measurement["code"] == VISIT_CODE:
                if len(features) == 0:
                    measurement["metadata"]["interpretation"] = "only_label"
                new_measurements.append(measurement)
                continue

            if len(features) > 0:
                if all(feature in codes_seen_today for feature in features):
                    if measurement["code"] in DISEASE_LABEL_CODES:
                        measurement["metadata"]["interpretation"] = "only_label"
                        new_measurements.append(measurement)
                    continue

                new_measurements.append(measurement)
                codes_seen_today |= set(features)
            else:
                # No match found --> check if code should still be included
                if measurement["code"] in DISEASE_LABEL_CODES:
                    copied_measurement = deepcopy(measurement)
                    copied_measurement["metadata"]["interpretation"] = "only_label"
                    new_measurements.append(copied_measurement)

                parent_codes = ontology.get_all_parents(measurement["code"])
                for parent_code in parent_codes:
                    # We don't want to add another visit code
                    if parent_code != VISIT_CODE:
                        measurement["code"] = parent_code
                        features, _ = tokenizer.get_feature_codes(
                            event["time"], measurement
                        )
                        if len(features) > 0:
                            if all(feature in codes_seen_today for feature in features):
                                continue
                            new_measurements.append(measurement)
                            codes_seen_today |= set(features)
                            # Only add the first match
                            break

        if len(new_measurements) > 0:
            if added_admission_id is not None:
                new_measurements = reorder_measurements(
                    new_measurements, added_admission_id
                )
            new_event: Event = {"measurements": new_measurements, "time": event["time"]}
            converted_patient["events"].append(new_event)
    return converted_patient
