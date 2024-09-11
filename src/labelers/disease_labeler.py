from femr.labelers import Labeler
from meds import Patient, Label
from typing import Literal, List
from datetime import datetime

from utility.utils import (
    get_admissions_with_last_measurement_date,
)
from utility.filter import filter_admissions
from utility.codes import VISIT_CODE


class DiseaseLabeler(Labeler):

    def __init__(
        self, codes: List[str], name: str, setting: Literal["end_of_stay", "next_year"]
    ) -> None:
        super().__init__()
        self.codes = set(codes)
        self.setting = setting
        self.name = name

    def get_previous_admission_end(self, admissions, admission_id):
        current_admission_start = admissions[admission_id]["start"]
        previous_admission_end = None

        for other_adm_id, admission in admissions.items():
            if other_adm_id == admission_id:
                continue

            other_adm_end = admission["end"]
            if other_adm_end <= current_admission_start:
                if (
                    previous_admission_end is None
                    or other_adm_end > previous_admission_end
                ):
                    previous_admission_end = other_adm_end

        return previous_admission_end

    def check_new_last_allowed_time(self, admissions, event_time):
        for admission_id, admission in admissions.items():
            if admission["start"] <= event_time <= admission["recorded_end"]:
                if event_time > admission["end"]:
                    admissions[admission_id]["end"] = event_time

    def get_admissions_with_diagnosis(self, patient: Patient):
        admissions = {}
        for event in patient["events"]:
            for measurement in event["measurements"]:
                admission_id = measurement["metadata"]["visit_id"]
                if measurement["code"] == VISIT_CODE:
                    # new admission
                    start = event["time"]
                    end = datetime.strptime(
                        measurement["metadata"]["end"], "%Y-%m-%d %H:%M:%S"
                    )
                    if admission_id not in admissions:
                        admissions[admission_id] = {
                            "start": start,
                            "end": start,
                            "code_match": False,
                            "recorded_end": end,
                        }
                    else:
                        admissions[admission_id]["start"] = start
                        admissions[admission_id]["end"] = start
                        admissions[admission_id]["recorded_end"] = end

                if (not "interpretation" in measurement["metadata"]) or measurement[
                    "metadata"
                ]["interpretation"] != "only_label":
                    self.check_new_last_allowed_time(admissions, event["time"])

                if measurement["code"] in self.codes:
                    if admission_id in admissions:
                        admissions[admission_id]["code_match"] = True
                    else:
                        admissions[admission_id] = {
                            "start": None,
                            "end": None,
                            "code_match": True,
                        }
        return admissions

    def label(self, patient: Patient) -> List[Label]:
        labels = []

        if self.setting == "end_of_stay":
            admissions = get_admissions_with_last_measurement_date(patient)
            filtered_admissions = filter_admissions(admissions)
            for admission_id in filtered_admissions:
                filtered_admissions[admission_id]["code_match"] = False
            for event in patient["events"]:
                for measurement in event["measurements"]:
                    if measurement["code"] in self.codes:
                        visit_id = measurement["metadata"]["visit_id"]
                        if visit_id in filtered_admissions:
                            filtered_admissions[visit_id]["code_match"] = True
            for admission_id in filtered_admissions:
                labels.append(
                    Label(
                        patient_id=patient["patient_id"],
                        prediction_time=filtered_admissions[admission_id][
                            "last_valid_measurement_time"
                        ],
                        boolean_value=filtered_admissions[admission_id]["code_match"],
                    )
                )
        else:
            assert self.setting == "next_year"
            admissions = self.get_admissions_with_diagnosis(patient)
            for admission_id, admission in admissions.items():
                previous_admission_end = self.get_previous_admission_end(
                    admissions, admission_id
                )
                if previous_admission_end is not None:
                    labels.append(
                        Label(
                            patient_id=patient["patient_id"],
                            prediction_time=previous_admission_end,
                            boolean_value=admission["code_match"],
                        )
                    )

        return labels

    def get_positive_codes(self):
        return list(self.codes)

    def __str__(self) -> str:
        return self.name + "_" + self.setting
