from femr.labelers import Labeler
from meds import Patient, Label
from typing import List
from random import Random
from datetime import timedelta

from utility.utils import get_admissions_with_last_measurement_date
from utility.filter import (
    filter_first_admission_of_day,
    filter_admission_type,
    choose_random_admission,
)
from utility.codes import BIASED_COHORT_SELECTION_CODES


class LongLOSLabeler(Labeler):
    """Labeler to label admissions positive at the end of the first day that have a stay longer than 7 days"""

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
        patient_id = patient["patient_id"]
        admissions = get_admissions_with_last_measurement_date(patient)
        admissions = filter_first_admission_of_day(admissions)

        if self.biased_admission_selection:
            admissions = filter_admission_type(
                admissions, BIASED_COHORT_SELECTION_CODES
            )

        if self.choose_random_admission:
            if len(admissions) > 1:
                admissions = filter_first_admission_of_day(admissions)
                if len(admissions) > 1:
                    admissions = choose_random_admission(
                        admissions, self.random_generator
                    )

        for admission_id in admissions:
            start = admissions[admission_id]["start"]
            end = admissions[admission_id]["end"]
            last_valid_measurement = admissions[admission_id][
                "last_valid_measurement_time"
            ]
            long_stay = (end - start) > timedelta(days=7)
            labels.append(
                Label(
                    patient_id=patient_id,
                    prediction_time=last_valid_measurement,
                    boolean_value=long_stay,
                )
            )
        return labels

    def __str__(self) -> str:
        labeler_repr = "long_los"
        if self.choose_random_admission:
            labeler_repr += "_rand_adm"
        if self.biased_admission_selection:
            labeler_repr += "_adm_selection_bias"
        return labeler_repr
