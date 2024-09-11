from utility.utils import get_admissions


def filter_admissions(admissions):
    """Exclude admissions that have death or discharge before prediction time."""
    filtered_admissions = {}
    for admission_id in admissions:
        prediction_time = admissions[admission_id]["start"].replace(
            hour=23, minute=59, second=59
        )
        if admissions[admission_id]["end"] > prediction_time:
            filtered_admissions[admission_id] = admissions[admission_id]
    return filtered_admissions


def filter_first_admission_of_day(admissions):
    """Exclude admissions that have death or discharge before prediction time and include only the first admission of a day."""
    filtered_admissions = {}
    first_admissions_by_day = {}

    for admission_id, admission_data in admissions.items():
        start_date = admission_data["start"].date()
        prediction_time = admission_data["start"].replace(hour=23, minute=59, second=59)

        if admission_data["end"] > prediction_time:
            if (
                start_date not in first_admissions_by_day
                or admission_data["start"]
                < first_admissions_by_day[start_date]["admission_data"]["start"]
            ):
                first_admissions_by_day[start_date] = {
                    "admission_id": admission_id,
                    "admission_data": admission_data,
                }

    for entry in first_admissions_by_day.values():
        filtered_admissions[entry["admission_id"]] = entry["admission_data"]

    return filtered_admissions


def choose_random_admission(admissions, random_generator):
    admission_id = random_generator.choice(list(admissions.keys()))
    return {admission_id: admissions[admission_id]}


def adm_filter_function(patient):
    admissions = get_admissions(patient)
    filtered_admissions = filter_admissions(admissions)
    return len(filtered_admissions) > 0


def adm_type_filter(patient):
    admissions = get_admissions(patient)
    admission_types = set()
    for admission in admissions.values():
        admission_types.add(admission["admission_type"])
    return "URGENT" in admission_types or "ELECTIVE" in admission_types


def filter_admission_type(admissions, include_types):
    result_admissions = {}
    for key, value in admissions.items():
        if value["admission_type"] in include_types:
            result_admissions[key] = value
    return result_admissions
