import sys
import os

# Adding the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

from src.split import PatientSplit

import argparse
from sklearn.metrics import roc_auc_score, average_precision_score
import pickle
import numpy as np
import pandas as pd


def bootstrap_confidence_interval(
    y_true, y_pred, metric_func, n_bootstraps=1000, ci=95
):
    """Calculate the confidence interval using bootstrapping."""
    bootstrapped_scores = []

    for i in range(n_bootstraps):
        # Resample the data with replacement
        indices = np.random.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            # Skip this sample if there are not at least two classes present
            continue

        score = metric_func(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)

    # Calculate the confidence interval
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    # Compute the lower and upper percentile based on the CI
    lower_bound = np.percentile(sorted_scores, (100 - ci) / 2)
    upper_bound = np.percentile(sorted_scores, 100 - (100 - ci) / 2)

    return lower_bound, upper_bound


def expected_calibration_error(y_true, y_pred, n_bins=10):
    """Calculate the Expected Calibration Error (ECE)."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_lowers = bin_edges[:-1]
    bin_uppers = bin_edges[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find indices where the predicted probability falls into the bin
        in_bin = np.logical_and(y_pred > bin_lower, y_pred <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_pred_in_bin = np.mean(y_pred[in_bin])
            ece += np.abs(avg_pred_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


def load_and_evaluate_model(model_path, X_test, y_test):
    """Load the model from the given path and evaluate it on the test set."""
    print(f"Loading model from {model_path}...")
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
            print(model)
            preds = model.predict_proba(X_test)[:, 1]
            return preds
    else:
        print(f"Model path {model_path} does not exist.")
        return None


def evaluate_task(task, task_features, split, output_path):
    """Evaluate a single task using Logistic Regression and XGBoost models."""
    print(f"Evaluating task: {task}")

    test_mask = np.isin(task_features["patient_ids"], split.test_patient_ids)
    X_test = task_features["features"][test_mask]
    y_test = task_features["boolean_values"][test_mask]

    models = {
        "Logistic Regression": os.path.join(
            output_path, "reg_model", f"{task}_model.pkl"
        ),
        "XGBoost": os.path.join(output_path, "xgb_model", f"{task}_model.pkl"),
    }

    results = []

    for model_name, model_path in models.items():
        preds = load_and_evaluate_model(model_path, X_test, y_test)
        if preds is None:
            continue

        if np.sum(y_test) == 0 or np.sum(y_test) == len(y_test):
            print(
                f"No positive or all positive examples in the test set for {model_name}, skipping metrics."
            )
            results.append([task, model_name, "N/A", "N/A", "N/A"])
        else:
            auroc = roc_auc_score(y_test, preds)
            auprc = average_precision_score(y_test, preds)
            ece = expected_calibration_error(y_test, preds)
            auroc_ci = bootstrap_confidence_interval(y_test, preds, roc_auc_score)
            auprc_ci = bootstrap_confidence_interval(
                y_test, preds, average_precision_score
            )

            results.append(
                [
                    task,
                    model_name,
                    f"{auroc:.4f} ({auroc_ci[0]:.4f} - {auroc_ci[1]:.4f})",
                    f"{auprc:.4f} ({auprc_ci[0]:.4f} - {auprc_ci[1]:.4f})",
                    f"{ece:.4f}",
                ]
            )
    return results


def save_results_to_latex(df, output_path):
    """Save the results dataframe to a LaTeX file."""
    latex_path = os.path.join(output_path, "results_table.tex")

    with open(latex_path, "w") as f:
        f.write(
            df.to_latex(
                index=False,
                column_format="|l|l|c|c|c|",
                longtable=True,
                escape=False,
            )
        )
    print(f"Results table saved as LaTeX file to {latex_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate results for downstream model"
    )
    parser.add_argument("--output-path", type=str)

    args = parser.parse_args()

    output_path = args.output_path
    features_path = os.path.join(output_path, "features")
    results_path = os.path.join(output_path, "results")
    os.makedirs(results_path, exist_ok=True)

    all_results = []

    with open(
        os.path.join(features_path, "features_and_labels.pkl"), "rb"
    ) as feature_file:
        features_and_labels = pickle.load(feature_file)

        for task, task_features in features_and_labels.items():

            split = PatientSplit.load_from_csv(
                os.path.join(output_path, "splits/split.csv")
            )
            task_results = evaluate_task(task, task_features, split, output_path)
            all_results.extend(task_results)

    # Create DataFrame for results
    df_results = pd.DataFrame(
        all_results,
        columns=["Task", "Model", "AUROC (95% CI)", "AUPRC (95% CI)", "ECE"],
    )

    # Print the table
    print(df_results)

    # Save results to text file
    with open(os.path.join(results_path, "results.txt"), "w") as results_file:
        df_results.to_string(results_file, index=False)

    # Save as LaTeX file
    save_results_to_latex(df_results, results_path)
