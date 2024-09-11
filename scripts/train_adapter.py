import sys
import os

# Adding the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

from src.split import PatientSplit

import argparse
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, precision_recall_curve, auc
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier


def load_features_and_labels(features_path):
    print(f"Loading features and labels from {features_path}...")
    with open(os.path.join(features_path, "features_and_labels.pkl"), "rb") as f:
        return pickle.load(f)


def load_patient_split(output_path):
    split_file = os.path.join(output_path, "splits/split.csv")
    print(f"Loading patient split from {split_file}...")
    return PatientSplit.load_from_csv(split_file)


def filter_data_by_mask(task_features, mask):
    return task_features["features"][mask], task_features["boolean_values"][mask]


def train_logistic_regression(
    X_train, y_train, X_val, y_val, class_weight_options, n_jobs
):
    print(f"Starting Logistic Regression training...")
    print(f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}")

    best_score = float("-inf")
    best_model = None
    best_l2 = 0
    best_weight_option = None
    for class_weight in class_weight_options:
        weight_desc = "unweighted" if class_weight is None else "weighted"
        print(f"Testing {weight_desc} variant.")
        for l2 in 10 ** np.linspace(1, -5, num=20):
            try:
                model = LogisticRegression(
                    C=l2, max_iter=10_000, n_jobs=n_jobs, class_weight=class_weight
                )
                model.fit(X_train, y_train)
                # Predict probabilities for the positive class
                y_proba = model.predict_proba(X_val)[:, 1]

                # Compute Precision-Recall curve
                precision, recall, _ = precision_recall_curve(y_val, y_proba)

                # Compute PR AUC
                pr_auc = auc(recall, precision)

                print(f"L2={l2}, PR AUC={pr_auc}, {weight_desc}")

                # Select the model with the highest PR AUC
                if pr_auc > best_score:
                    best_score = pr_auc
                    best_model = model
                    best_l2 = l2
                    best_weight_option = weight_desc
                    print(
                        f"New best model found with L2={best_l2}, PR AUC={best_score}, {weight_desc}"
                    )
            except Exception as e:
                print(f"Error during Logistic Regression training: {e}")

    return best_model, best_score, best_l2, best_weight_option


def train_xgboost(X_train, y_train, X_val, y_val, scale_pos_weight_options, n_jobs):
    print(f"Starting XGBoost training...")
    print(f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}")

    best_score = float("-inf")
    best_model = None
    best_lr = 0
    best_weight_option = None
    for scale_pos_weight in scale_pos_weight_options:
        weight_desc = "unweighted" if scale_pos_weight == 1 else "weighted"
        print(f"Testing {weight_desc} variant.")
        for lr in 10 ** np.linspace(0, -3, num=10):
            try:
                model = XGBClassifier(
                    learning_rate=lr,
                    n_estimators=100,
                    max_depth=5,
                    objective="binary:logistic",
                    n_jobs=n_jobs,
                    scale_pos_weight=scale_pos_weight,
                )
                model.fit(X_train, y_train)
                # Predict probabilities for the positive class
                y_proba = model.predict_proba(X_val)[:, 1]

                # Compute Precision-Recall curve
                precision, recall, _ = precision_recall_curve(y_val, y_proba)

                # Compute PR AUC
                pr_auc = auc(recall, precision)

                print(f"Learning Rate={lr}, PR AUC={pr_auc}, {weight_desc}")

                # Select the model with the highest PR AUC
                if pr_auc > best_score:
                    best_score = pr_auc
                    best_model = model
                    best_lr = lr
                    best_weight_option = weight_desc
                    print(
                        f"New best model found with Learning Rate={best_lr}, PR AUC={best_score}, {weight_desc}"
                    )
            except Exception as e:
                print(f"Error during XGBoost training: {e}")

    return best_model, best_score, best_lr, best_weight_option


def save_model_and_results(
    output_dir, task, model, score, hyperparam_name, hyperparam_value, weight_desc
):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving model and results to {output_dir}...")

    results_file = os.path.join(output_dir, f"{task}_best_model_results.txt")
    with open(results_file, "w") as file:
        file.write(f"Best Score: {score}\n")
        file.write(f"Best {hyperparam_name}: {hyperparam_value}\n")
        file.write(f"Weighting: {weight_desc}\n")
    print(f"Results saved to {results_file}.")

    model_file = os.path.join(output_dir, f"{task}_model.pkl")
    with open(model_file, "wb") as file:
        pickle.dump(model, file)
    print(f"Model saved to {model_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=str)
    parser.add_argument("--n-procs", type=int, default=4)

    args = parser.parse_args()

    print("Train adapter")
    output_path = args.output_path
    n_jobs = args.n_procs

    features_path = os.path.join(output_path, "features")
    features_and_labels = load_features_and_labels(features_path)

    for task, task_features in features_and_labels.items():
        print(f"\n\nProcessing task: {task}")

        split = load_patient_split(output_path)
        train_mask = np.isin(task_features["patient_ids"], split.train_patient_ids)
        val_mask = np.isin(task_features["patient_ids"], split.val_patient_ids)

        X_train, y_train = filter_data_by_mask(task_features, train_mask)
        X_val, y_val = filter_data_by_mask(task_features, val_mask)

        # Check if there are positive and negative labels in the training data
        if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
            print(
                f"Skipping task {task} due to lack of positive/negative labels in the training data."
            )
            continue

        class_weight = compute_class_weight(
            class_weight="balanced", classes=np.unique(y_train), y=y_train
        )
        class_weight_dict = {i: class_weight[i] for i in range(len(class_weight))}

        class_weight_options = [None, class_weight_dict]
        best_model_lr, best_score_lr, best_l2_lr, best_weight_lr = (
            train_logistic_regression(
                X_train, y_train, X_val, y_val, class_weight_options, n_jobs
            )
        )
        save_model_and_results(
            os.path.join(output_path, "reg_model"),
            task,
            best_model_lr,
            best_score_lr,
            "L2 Regularization",
            best_l2_lr,
            best_weight_lr,
        )

        scale_pos_weight_options = [
            1,
            class_weight_dict[1] / class_weight_dict[0],
        ]  # Unweighted is equivalent to scale_pos_weight=1
        best_model_xgb, best_score_xgb, best_lr_xgb, best_weight_xgb = train_xgboost(
            X_train, y_train, X_val, y_val, scale_pos_weight_options, n_jobs
        )
        save_model_and_results(
            os.path.join(output_path, "xgb_model"),
            task,
            best_model_xgb,
            best_score_xgb,
            "Learning Rate",
            best_lr_xgb,
            best_weight_xgb,
        )
