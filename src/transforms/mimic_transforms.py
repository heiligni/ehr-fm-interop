from meds import Measurement, Patient
from femr.models.tokenizer import FEMRTokenizer
from typing import List
from collections import deque


"""This function is needed to make sure the order of measurements is right.
Sometimes the last measurements of an old visit have the same timestamps as measurements of 
new visits. Then, in order to generate correct labels it is important that measurements of new visits
come after measurements of the older visit
"""


def reorder_measurements(measurements: List[Measurement], visit_id: str):
    new_measurements = deque([])
    for measurement in measurements:
        if (
            measurement["metadata"]["visit_id"] is not None
            and measurement["metadata"]["visit_id"] != visit_id
        ):
            new_measurements.appendleft(measurement)
        else:
            new_measurements.append(measurement)
    return list(new_measurements)


def tokenizer_has_code(tokenizer: FEMRTokenizer, code: str, type: str, numeric_value):
    if type == "numeric":
        assert numeric_value is not None
        for start, end, _ in tokenizer.numeric_lookup.get(code, []):
            if start <= numeric_value < end:
                return True
    elif type == "text":
        if code in tokenizer.string_lookup:
            return True
    else:
        if code in tokenizer.code_lookup:
            return True
    return False


def evaluate_measurement(numeric_value, lower_bound, upper_bound):
    # Type conversion
    if type(lower_bound) == str:
        lower_bound = float(lower_bound)
    if type(upper_bound) == str:
        upper_bound = float(upper_bound)

    if lower_bound is not None and upper_bound is not None:
        if numeric_value < lower_bound:
            return "decreased"
        elif numeric_value >= lower_bound and numeric_value <= upper_bound:
            return "normal"
        elif numeric_value > upper_bound:
            return "increased"
        else:
            raise ValueError("Invalid numeric value")
    if lower_bound is not None:
        if numeric_value < lower_bound:
            return "decreased"
        elif numeric_value >= lower_bound:
            return "normal"
        else:
            raise ValueError("Invalid numeric value")
    if upper_bound is not None:
        if numeric_value <= upper_bound:
            return "normal"
        elif numeric_value > upper_bound:
            return "increased"
        else:
            raise ValueError("Invalid numeric value")
    raise ValueError("Invalid parameters")


def lab_measurement_evaluation_batched(batch):
    for events in batch["events"]:
        for event in events:
            for measurement in event["measurements"]:
                if measurement["numeric_value"]:
                    metadata = measurement["metadata"]
                    lower_bound, upper_bound = None, None
                    if metadata["ref_range_lower"] is not None:
                        lower_bound = metadata["ref_range_lower"]
                    if metadata["ref_range_upper"] is not None:
                        upper_bound = metadata["ref_range_upper"]
                    if lower_bound is not None or upper_bound is not None:
                        evaluation = evaluate_measurement(
                            measurement["numeric_value"], lower_bound, upper_bound
                        )
                        measurement["text_value"] = evaluation
    return batch
