from meds import Patient, Label, death_code, Event
from typing import List
from femr.labelers import Labeler
from random import Random

from utility.utils import get_admissions_with_last_measurement_date
from utility.filter import (
    filter_first_admission_of_day,
    choose_random_admission,
    filter_admission_type,
)
from utility.codes import BIASED_COHORT_SELECTION_CODES


class MortalityLabeler(Labeler):

    def __init__(
        self,
        choose_random_admission=False,
        biased_admission_selection=False,
        seed=12345,
    ) -> None:
        super().__init__()
        self.choose_random_admission = choose_random_admission
        self.biased_admission_selection = biased_admission_selection
        self.random_generator = Random(seed)

    def label(self, patient: Patient) -> List[Label]:
        labels = []
        admissions = get_admissions_with_last_measurement_date(patient)
        admissions = filter_first_admission_of_day(admissions)

        if self.biased_admission_selection:
            admissions = filter_admission_type(
                admissions, BIASED_COHORT_SELECTION_CODES
            )

        if self.choose_random_admission:
            if len(admissions) > 1:
                admissions = choose_random_admission(admissions, self.random_generator)

        for event in patient["events"]:
            for measurement in event["measurements"]:
                if measurement["code"] == death_code:
                    death_time = event["time"]
                    for admission_id, admission_values in admissions.items():
                        if (
                            admission_values["start"] <= death_time
                            and admission_values["end"].date() >= death_time.date()
                        ):
                            admissions[admission_id]["death"] = True

        for admission in admissions.values():
            death = True if "death" in admission else False
            labels.append(
                Label(
                    patient_id=patient["patient_id"],
                    prediction_time=admission["last_valid_measurement_time"],
                    boolean_value=death,
                )
            )

        return labels

    def __str__(self) -> str:
        labeler_repr = "mortality"
        if self.choose_random_admission:
            labeler_repr += "_rand_adm"
        if self.biased_admission_selection:
            labeler_repr += "_adm_selection_bias"
        return labeler_repr
