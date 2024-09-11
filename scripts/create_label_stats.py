import pandas as pd
import os
import argparse


def count_boolean_values(df):
    """
    Count the number of True and False boolean values in the given DataFrame.
    :param df: The DataFrame containing boolean values.
    :return: A tuple (num_true, num_false) where num_true is the count of True values and num_false is the count of False values.
    """
    num_true = df["boolean_value"].sum()
    num_false = len(df) - num_true
    return num_true, num_false


def calculate_avg_labels_per_patient(df):
    """
    Calculate the average number of labels per patient and the average number of True labels per patient.
    :param df: The DataFrame containing boolean values and patient IDs.
    :return: A tuple with the average number of labels per patient and the average number of True labels per patient.
    """
    # Calculate total labels per patient
    labels_per_patient = df.groupby("patient_id").size()
    avg_labels = labels_per_patient.mean()

    # Calculate True labels per patient, filling missing patients with 0
    true_labels_per_patient = (
        df[df["boolean_value"] == True].groupby("patient_id").size()
    )
    true_labels_per_patient = true_labels_per_patient.reindex(
        labels_per_patient.index, fill_value=0
    )
    avg_true_labels = true_labels_per_patient.mean()

    return avg_labels, avg_true_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run label statistics")
    parser.add_argument("--output-path", type=str)

    args = parser.parse_args()

    print("Starting label stats generation")

    label_dir = os.path.join(args.output_path, "labels")
    output_file = os.path.join(label_dir, "label_counts.txt")

    total_files = len(
        [filename for filename in os.listdir(label_dir) if filename.endswith(".csv")]
    )
    processed_files = 0

    with open(output_file, "w") as f_out:
        f_out.write(
            "{:<43} {:<22} {:<22} {:<21} {:<21} {:<25} {:<30}\n".format(
                "Filename",
                "Number of True values",
                "Number of False values",
                "True Percentage",
                "False Percentage",
                "Avg Labels per Patient",
                "Avg True Labels per Patient",
            )
        )

        for filename in os.listdir(label_dir):
            if filename.endswith(".csv"):
                file_path = os.path.join(label_dir, filename)
                try:
                    df = pd.read_csv(file_path)

                    num_true, num_false = count_boolean_values(df)
                    total = num_true + num_false
                    percentage_true = (num_true / total) * 100
                    percentage_false = (num_false / total) * 100

                    avg_labels_per_patient, avg_true_labels_per_patient = (
                        calculate_avg_labels_per_patient(df)
                    )

                    f_out.write(
                        "{:<43} {:<22} {:<22} {:<20.4f}% {:<20.4f}% {:<25.2f} {:<30.2f}\n".format(
                            filename,
                            num_true,
                            num_false,
                            percentage_true,
                            percentage_false,
                            avg_labels_per_patient,
                            avg_true_labels_per_patient,
                        )
                    )

                    print(
                        f"Processed {filename}: {num_true} True, {num_false} False, {percentage_true:.2f}% True, {percentage_false:.2f}% False, Avg labels/patient: {avg_labels_per_patient:.2f}, Avg true labels/patient: {avg_true_labels_per_patient:.2f}"
                    )

                except Exception as e:
                    print(f"Failed to process {filename}: {e}")

                processed_files += 1
                print(f"Progress: {processed_files}/{total_files} files processed")

    print(f"All results have been written to {output_file}")
