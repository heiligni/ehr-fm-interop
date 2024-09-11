import numpy as np
from texttable import Texttable
import latextable
from utility.file_utils import create_if_not_exists
import os
from utility.config import cfg

# Function to calculate min, mean, and max
def calculate_statistics(data):
    return {
        "min": np.min(data),
        "mean": np.mean(data),
        "max": np.max(data),
        "first_quartile": np.percentile(data, 25),
        "third_quartile": np.percentile(data, 75),
        "total": np.sum(data)
    }

# Function to format days to "xyz days (x.y years)"
def format_timedelta(timedelta):
    days = timedelta.days
    hours = timedelta.seconds / 3600
    days += hours / 24
    years = days / 365
    return f"{days:,.0f} days ({years:,.1f} years)"

def create_code_stat_table(summary_ds, stage):
    # Calculate statistics for each split and all splits combined
    stats = {}

    stats = {}
    for attr in summary_ds.column_names:
        stats[attr] = calculate_statistics(summary_ds[attr])

    # Generate the table
    rows = [["Attribute", ""]]

    rows.append(["total patients", len(summary_ds)])

    for attr in ["events_per_patient", "visits_per_patient", "timeline_length_per_patient"]:
        rows.append([f"Number of {attr.split('_')[0].capitalize()}", ""])
        for metric in stats[attr].keys():
            if attr == "timeline_length_per_patient":
                row = [
                    metric.capitalize(),
                    format_timedelta(stats[attr][metric]),
                ]
            else:
                row = [
                    metric.capitalize(),
                    stats[attr][metric],
                ]
            rows.append(row)

    table = Texttable()
    table.set_cols_align(["l", "c"])
    table.set_deco(Texttable.HEADER | Texttable.VLINES)
    table.add_rows(rows)
    print(table.draw())


    output_dir_path = os.path.join(cfg["data"]["data_dir"], cfg["data"]["table_dir"])
    create_if_not_exists(output_dir_path)
    # Save the LaTeX table
    print(f"Saving table to {output_dir_path}")
    with open(os.path.join(output_dir_path,  f"summary_stats_{stage}.txt"), 'w') as f:
        latex_txt = latextable.draw_latex(table, caption="Number of inputs across data splits")
        f.write(latex_txt)