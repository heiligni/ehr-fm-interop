"""OMOP CDM Codes"""

VISIT_CODE = "Visit/IP"


"""MIMIC Codes"""
ADMISSION_PREFIX = "MIMIC_IV_Admission/"  # followed by admission type
RACE_CODE = "MIMIC_IV_Race/"  # followed by race column value
ICD9_PREFIX = "ICD9CM/"
ICD10_PREFIX = "ICD10CM/"

ADMISSION_TYPES = [
    "AMBULATORY OBSERVATION",
    "SURGICAL SAME DAY ADMISSION",
    "ELECTIVE",
    "DIRECT OBSERVATION",
    "OBSERVATION ADMIT",
    "URGENT",
    "EU OBSERVATION",
    "EW EMER.",
    "DIRECT EMER.",
]

ADMISSION_CODES = [ADMISSION_PREFIX + adm_type for adm_type in ADMISSION_TYPES]


BIASED_COHORT_SELECTION_CODES = [
    "ELECTIVE",
    "URGENT",
]

AKI_CODES = set(
    [
        "ICD10CM/N14.11",
        "ICD10CM/N17",
        "ICD10CM/N17.1",
        "ICD10CM/N17.2",
        "ICD10CM/N17.8",
        "ICD10CM/N17.9",
        "ICD10CM/O90.4",
        "ICD10CM/O90.49",
        "ICD9CM/584",
        "ICD9CM/584.5",
        "ICD9CM/584.6",
        "ICD9CM/584.7",
        "ICD9CM/584.8",
        "ICD9CM/584.9",
        "ICD9CM/669.3",
        "ICD9CM/669.30",
        "ICD9CM/669.32",
        "ICD9CM/669.34",
    ]
)

HYPERLIPIDAEMIA_CODES = set(
    [
        "ICD10CM/E78.0",
        "ICD10CM/E78.00",
        "ICD10CM/E78.01",
        "ICD10CM/E78.1",
        "ICD10CM/E78.2",
        "ICD10CM/E78.3",
        "ICD10CM/E78.4",
        "ICD10CM/E78.49",
        "ICD10CM/E78.5",
        "ICD9CM/272.0",
        "ICD9CM/272.1",
        "ICD9CM/272.2",
        "ICD9CM/272.3",
        "ICD9CM/272.4",
    ]
)

CKD_CODES = set(
    [
        "ICD10CM/E08.22",
        "ICD10CM/E11.22",
        "ICD10CM/I12",
        "ICD10CM/I12.9",
        "ICD10CM/I13",
        "ICD10CM/I13.1",
        "ICD10CM/I13.10",
        "ICD10CM/I13.11",
        "ICD10CM/I13.2",
        "ICD10CM/N18",
        "ICD10CM/N18.1",
        "ICD10CM/N18.2",
        "ICD10CM/N18.3",
        "ICD10CM/N18.30",
        "ICD10CM/N18.31",
        "ICD10CM/N18.32",
        "ICD10CM/N18.4",
        "ICD10CM/N18.5",
        "ICD10CM/N18.6",
        "ICD10CM/N18.9",
        "ICD10CM/N25.0",
        "ICD10CM/O10.21",
        "ICD10CM/O10.211",
        "ICD10CM/O10.212",
        "ICD10CM/O10.213",
        "ICD10CM/O10.219",
        "ICD10CM/O10.22",
        "ICD10CM/O10.23",
        "ICD10CM/O10.31",
        "ICD10CM/O10.311",
        "ICD10CM/O10.312",
        "ICD10CM/O10.313",
        "ICD10CM/O10.319",
        "ICD10CM/O10.32",
        "ICD10CM/O10.33",
        "ICD9CM/403",
        "ICD9CM/403.00",
        "ICD9CM/403.01",
        "ICD9CM/403.91",
        "ICD9CM/404",
        "ICD9CM/404.00",
        "ICD9CM/404.02",
        "ICD9CM/404.03",
        "ICD9CM/404.10",
        "ICD9CM/404.90",
        "ICD9CM/404.92",
        "ICD9CM/404.93",
        "ICD9CM/585",
        "ICD9CM/585.1",
        "ICD9CM/585.2",
        "ICD9CM/585.3",
        "ICD9CM/585.4",
        "ICD9CM/585.5",
        "ICD9CM/585.6",
        "ICD9CM/585.9",
        "ICD9CM/588.0",
    ]
)
