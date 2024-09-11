from datetime import datetime, timedelta
from typing import Callable, Sequence
import functools
from datasets import Dataset
from meds import Patient, death_code, Event, Measurement
from femr.pat_utils import get_patient_birthdate

from utility.dates import parse_date
from utility.codes import ADMISSION_PREFIX
from cleaning.clean_rules import rules
import re


def change_visits(patient: Patient, visit_changes) -> Patient:
    new_events: list[Event] = []
    for event in patient["events"]:
        new_measurements: list[Measurement] = []
        for measurement in event["measurements"]:
            visit_id = measurement["metadata"]["visit_id"]
            if visit_id in visit_changes:
                if measurement["code"].startswith(ADMISSION_PREFIX):
                    # code for admission, if an admission is assigned a new id, the start code for the old admission is removed from the sequence
                    if not "new_id" in visit_changes[visit_id]:
                        assert "new_end" in visit_changes[visit_id]
                        measurement["metadata"]["end"] = str(
                            visit_changes[visit_id]["new_end"]
                        )
                        new_measurements.append(measurement)
                else:
                    # for non-admission measurement we have to change the visit id
                    if "new_id" in visit_changes[visit_id]:
                        measurement["metadata"]["visit_id"] = visit_changes[visit_id][
                            "new_id"
                        ]
                    new_measurements.append(measurement)
            else:
                new_measurements.append(measurement)
        if len(new_measurements) > 0:
            new_events.append({"time": event["time"], "measurements": new_measurements})
    return {
        "patient_id": patient["patient_id"],
        "static_measurements": [],
        "events": new_events,
    }


def fit_codes_to_visit_times(patient: Patient) -> Patient:
    visit_times = extract_visit_times(patient)

    visit_map = {
        visit["visit_id"]: {"start": visit["start"], "end": visit["end"]}
        for visit in visit_times
    }

    patient = assign_codes_to_visit_end(patient, visit_map)
    rearrange_events(patient)

    return patient


def assign_codes_to_visit_end(patient: Patient, visit_times) -> Patient:
    insert_at_new_time = []
    new_events = []
    for event in patient["events"]:
        new_measurements = []
        for measurement in event["measurements"]:
            visit_id = measurement["metadata"]["visit_id"]
            if visit_id is not None:
                if event["time"] < visit_times[visit_id]["start"]:
                    insert_at_new_time.append(
                        {
                            "time": visit_times[visit_id]["start"],
                            "measurement": measurement,
                        }
                    )
                elif event["time"] > visit_times[visit_id]["end"]:
                    insert_at_new_time.append(
                        {
                            "time": visit_times[visit_id]["end"],
                            "measurement": measurement,
                        }
                    )
                else:
                    new_measurements.append(measurement)
            else:
                new_measurements.append(measurement)
        if len(new_measurements) > 0:
            new_events.append({"time": event["time"], "measurements": new_measurements})
    insert_at_new_time.sort(key=lambda x: x["time"])

    current_idx = 0
    while len(insert_at_new_time) > 0:
        insert_event = insert_at_new_time.pop(0)
        insert_event_time = insert_event["time"]
        event_time = new_events[current_idx]["time"]
        if event_time > insert_event_time:
            current_idx += 1
        elif event_time == insert_event_time:
            new_events[current_idx]["measurements"].append(insert_event["measurement"])
        else:
            new_events.append(
                {
                    "time": insert_event_time,
                    "measurements": [insert_event["measurement"]],
                }
            )

    return {
        "patient_id": patient["patient_id"],
        "static_measurements": [],
        "events": new_events,
    }


def extract_visit_times(patient):
    visits = []
    events = patient["events"]
    for event in events:
        for measurement in event["measurements"]:
            visit_id = measurement["metadata"]["visit_id"]
            if measurement["code"].startswith(ADMISSION_PREFIX):
                end = measurement["metadata"]["end"]
                if isinstance(end, str):
                    end = parse_date(end)
                if end is None:
                    end = events[len(events) - 1]["time"]
                visits.append(
                    {"visit_id": visit_id, "start": event["time"], "end": end}
                )
    return visits


def find_overlapping_visits(visits):
    sorted_visits = sorted(visits, key=lambda item: item["start"])

    current_end = sorted_visits[0]["end"]
    current_visit_id = sorted_visits[0]["visit_id"]

    visits_to_reassign = {}

    for visit in sorted_visits[1:]:
        if current_end + timedelta(minutes=15) >= visit["start"]:
            visits_to_reassign[visit["visit_id"]] = {
                "new_id": current_visit_id,
            }
            # If we merge overlapping visits, we potentially have to correct the end of the earlier visit
            if visit["end"] > current_end:
                visits_to_reassign[current_visit_id] = {"new_end": visit["end"]}
                current_end = visit["end"]
        else:
            current_end = visit["end"]
            current_visit_id = visit["visit_id"]

    return visits_to_reassign


def merge_overlapping_visits(patient: Patient) -> Patient:
    visits = extract_visit_times(patient)
    if len(visits) > 0:
        overlapping_visits = find_overlapping_visits(visits)
        if len(overlapping_visits) > 0:
            return change_visits(patient, overlapping_visits)
    return patient


def move_death_to_end(patient: Patient):
    death_measurement = None
    new_patient: Patient = {
        "events": [],
        "patient_id": patient["patient_id"],
        "static_measurements": [],
    }
    for event in patient["events"]:
        new_measurements = []
        for measurement in event["measurements"]:
            if measurement["code"] != death_code:
                new_measurements.append(measurement)
            else:
                death_measurement = (event["time"], measurement)
        if len(new_measurements) > 0:
            new_patient["events"].append(
                {"time": event["time"], "measurements": new_measurements}
            )
    if death_measurement is not None:
        time, measurement = death_measurement
        last_event = new_patient["events"][len(new_patient["events"]) - 1]
        last_time = last_event["time"]
        if time <= last_time:
            last_event["measurements"].append(measurement)
        else:
            new_patient["events"].append({"measurements": [measurement], "time": time})
    return new_patient


def clean_redacted_texts(measurement: Measurement):
    if measurement["text_value"] is not None:
        if re.match(r"^_+$", measurement["text_value"]):
            measurement["text_value"] = None


def normalize_codes(patient: Patient) -> Patient:
    for event in patient["events"]:
        new_measurements = []
        for measurement in event["measurements"]:
            clean_redacted_texts(measurement)

            remove = False
            for rule in rules:
                if (
                    rule["match_type"] == "full_match"
                    and rule["code"] == measurement["code"]
                ) or (
                    rule["match_type"] == "start"
                    and measurement["code"].startswith(rule["code"])
                ):
                    if "remove" in rule and rule["remove"]:
                        remove = True
                    elif rule["replace"] is not None:
                        measurement["code"] = rule["replace"]
            if not remove:
                new_measurements.append(measurement)
        event["measurements"] = new_measurements
    return patient


def rearrange_events(patient: Patient):
    patient["events"].sort(key=lambda a: a["time"])


def move_pre_birth(patient: Patient) -> Patient:
    """Move all events to after the birth of a patient."""
    birth_date = get_patient_birthdate(patient)

    for event in patient["events"]:
        if event["time"] < birth_date:
            delta = birth_date - event["time"]
            if delta > datetime.timedelta(days=30):
                continue

            event["time"] = birth_date

            for measurement in event["measurements"]:
                if (
                    measurement["metadata"].get("end") is not None
                    and measurement["metadata"]["end"] < birth_date
                ):
                    measurement["metadata"]["end"] = birth_date
    rearrange_events(patient)
    return patient


def get_mimic_transformations():
    """Get the list of MIMIC-IV transformations"""
    transforms: Sequence[Callable[[Patient], Patient]] = [
        move_pre_birth,
        # move_death_to_end,
        normalize_codes,
        merge_overlapping_visits,
        fit_codes_to_visit_times,
    ]

    return lambda patient: functools.reduce(lambda r, f: f(r), transforms, patient)


def clean_patients(dataset: Dataset, num_proc: int) -> Dataset:
    # Process did die with higher amount of num_proc than 1 locally
    print(f"Clean patients with {num_proc} procs")
    return dataset.map(
        get_mimic_transformations(), num_proc=num_proc, load_from_cache_file=False
    )
