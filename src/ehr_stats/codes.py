from collections import defaultdict
from texttable import Texttable
import latextable
from statistics import mean
import os
import matplotlib.pyplot as plt
from utility.utils import combine_stats


def extract_code_stats(batch):
    codes = defaultdict(int)
    total_code_count = []
    unique_code_count = []
    for events in batch["events"]:
        total_codes_patient = 0
        unique_codes = set()
        for event in events:
            for measurement in event["measurements"]:
                total_codes_patient += 1
                unique_codes.add(measurement["code"])
                codes[measurement["code"]] += 1
        total_code_count.append(total_codes_patient)
        unique_code_count.append(len(unique_codes))
    return {
        "codes": dict(codes),
        "total_code_count": total_code_count,
        "unique_code_count": unique_code_count,
    }


def combine_code_stats(stats1, stats2):
    codes = combine_stats(stats1["codes"], stats2["codes"])
    if stats1["total_code_count"] is None:
        stats1["total_code_count"] = []
    if stats1["unique_code_count"] is None:
        stats1["unique_code_count"] = []

    total_code_count = stats1["total_code_count"]
    unique_code_count = stats1["unique_code_count"]

    total_code_count.extend(stats2["total_code_count"])
    unique_code_count.extend(stats2["unique_code_count"])

    return {
        "codes": codes,
        "total_code_count": total_code_count,
        "unique_code_count": unique_code_count,
    }


def create_code_stats_table(stats, output_dir_path):
    code_table = Texttable()
    code_table.set_cols_align(["l"] + ["c"] * 2)
    code_table.set_deco(Texttable.HEADER | Texttable.VLINES)
    code_table.add_row(["", "Total Codes per Patient", "Unique Codes per Patient"])
    code_table.add_row(
        ["min", min(stats["total_code_count"]), min(stats["unique_code_count"])]
    )
    code_table.add_row(
        ["mean", mean(stats["total_code_count"]), mean(stats["unique_code_count"])]
    )
    code_table.add_row(
        ["max", max(stats["total_code_count"]), max(stats["unique_code_count"])]
    )

    total_codes = sum(stats["total_code_count"])

    print(code_table.draw())
    print("Total Events: ", total_codes)
    os.makedirs(output_dir_path, exist_ok=True)
    with open(os.path.join(output_dir_path, "code_stats.txt"), "w") as f:
        latex_txt = latextable.draw_latex(
            code_table, caption="Code statistics before and after Preprocessing."
        )
        f.write(latex_txt)
        f.write(f"\nTotal Events: {total_codes}")


def plot_codes_per_patient(unique_code_counts, plot_dir_path):
    plt.figure(figsize=(10, 6))
    plt.hist(unique_code_counts, bins=50, edgecolor="black")
    plt.xlabel("Number of unique Codes")
    plt.ylabel("Number of Patients")
    plt.savefig(os.path.join(plot_dir_path, "unique_codes_hist.png"))


def create_code_occurence_plot(code_stats, plot_dir_path):
    plt.figure(figsize=(10, 6))
    plt.hist(code_stats["codes"].values(), bins=50, edgecolor="black")
    plt.xlabel("Total Occurences in Patient Timelines")
    plt.ylabel("Amount of Codes")
    plt.savefig(os.path.join(plot_dir_path, "code_occurence_hist.png"))


def get_summary_statistics(batch):
    events_per_patient = []
    visits_per_patient = []
    timeline_length_per_patient = []
    for events in batch["events"]:
        if len(events) == 0:
            # Handle cases where there are no events for a patient
            events_per_patient.append(0)
            visits_per_patient.append(0)
            timeline_length_per_patient.append(0)
            continue

        visit_ids = set()
        total_events = 0
        if len(events) > 1:
            first_event_date = events[1]["time"]
        else:
            first_event_date = events[0]["time"]
        last_event_date = events[-1]["time"]

        for event in events:
            for measurement in event["measurements"]:
                total_events += 1
                if "visit_id" in measurement["metadata"]:
                    visit_ids.add(measurement["metadata"]["visit_id"])

        events_per_patient.append(total_events)
        timeline_length_per_patient.append(last_event_date - first_event_date)
        visits_per_patient.append(len(visit_ids))

    return {
        "events_per_patient": events_per_patient,
        "visits_per_patient": visits_per_patient,
        "timeline_length_per_patient": timeline_length_per_patient,
    }
