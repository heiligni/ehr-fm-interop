from collections import defaultdict
from meds import Event
from meds import Patient, birth_code
from datetime import datetime

from utility.codes import VISIT_CODE, ADMISSION_PREFIX, ADMISSION_CODES
from utility.dates import parse_date


def _add_total(row):
    for i in range(1, 4):
        if row[i] is None:
            row[i] = 0
    row[4] = row[1] + row[2] + row[3]


def combine_stats(stats1: dict, stats2: dict) -> dict:
    result = defaultdict(int)

    for key, value in stats1.items():
        result[key] += value

    for key, value in stats2.items():
        result[key] += value

    return dict(result)


def combine_sets(set1: set, set2: set) -> set:
    return set1.union(set2)


def get_measurement(events: list[Event], code: str):
    for event in events:
        for measurement in event["measurements"]:
            if measurement["code"].startswith(code):
                return measurement
    return None


def get_admissions_with_last_measurement_date(patient: Patient):
    admissions = {}
    codes_seen_today = set()
    current_date = None
    last_valid_measurement_time = None
    current_visit = None

    for event in patient["events"]:
        event_date = event["time"].date()

        if event_date != current_date:
            current_date = event_date
            codes_seen_today = set()
            last_valid_measurement_time = event["time"]

        for measurement in event["measurements"]:
            visit_id = measurement["metadata"]["visit_id"]

            if measurement["code"] == VISIT_CODE:
                # new admission
                start = event["time"]
                end = parse_date(measurement["metadata"]["end"])
                admission_type = measurement["metadata"]["priority"]
                visit = {
                    "start": start,
                    "end": end,
                    "last_valid_measurement_time": last_valid_measurement_time,
                    "last_code": measurement["code"],
                    "admission_type": admission_type,
                }
                admissions[visit_id] = visit
                current_visit = visit
            elif measurement["code"] not in codes_seen_today and not (
                "interpretation" in measurement["metadata"]
                and measurement["metadata"]["interpretation"] == "only_label"
            ):
                if (
                    visit_id is not None
                    and visit_id in admissions
                    and event["time"]
                    <= admissions[visit_id]["start"].replace(
                        hour=23, minute=59, second=59
                    )
                ):
                    admissions[visit_id]["last_valid_measurement_time"] = event["time"]
                    admissions[visit_id]["last_code"] = measurement["code"]
                # Handle cases where the visit code is not present
                elif (
                    current_visit is not None
                    and event["time"]
                    <= current_visit["start"].replace(hour=23, minute=59, second=59)
                    and current_visit["start"] <= event["time"]
                ):
                    current_visit["last_valid_measurement_time"] = event["time"]
                    current_visit["last_code"] = measurement["code"]

            codes_seen_today.add(measurement["code"])

    filtered_admissions = {}

    # If an admission has no valid code at all it should not be included
    for admission_id, details in admissions.items():
        start_time = details.get("start")
        last_valid_time = details.get("last_valid_measurement_time")

        # Check if the last valid measurement time is before the start time
        if last_valid_time >= start_time:
            filtered_admissions[admission_id] = details

    return filtered_admissions


def get_admissions(patient: Patient):
    admissions = {}
    for event in patient["events"]:
        for measurement in event["measurements"]:
            if measurement["code"] == VISIT_CODE or measurement["code"].startswith(
                ADMISSION_PREFIX
            ):
                # new admission
                admission_id = measurement["metadata"]["visit_id"]
                start = event["time"]
                end = datetime.strptime(
                    measurement["metadata"]["end"], "%Y-%m-%d %H:%M:%S"
                )
                admissions[admission_id] = {"start": start, "end": end}
                if measurement["code"].startswith(ADMISSION_PREFIX):
                    admission_type = measurement["code"].split("/")[1]
                    admissions[admission_id]["admission_type"] = admission_type
    return admissions


def get_measurement_time(events: list[Event], code: str):
    for event in events:
        for measurement in event["measurements"]:
            if measurement["code"].startswith(code):
                return event["time"]
    return None


def get_anchor_age_batched(batch: list[Patient]) -> datetime:
    result = []
    for events in batch["events"]:
        birth_measurement = get_measurement(events, birth_code)
        if birth_measurement is not None:
            result.append(
                int(birth_measurement["metadata"]["anchor_information"].split("/")[1])
            )
        else:
            raise ValueError("Couldn't find patient birthdate -- Patient has no events")
    return result
