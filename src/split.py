import csv
import dataclasses
from typing import List
from datasets import Dataset, DatasetDict
import femr.index
import femr.splits


def create_and_save_split(dataset: Dataset, frac_test: float, frac_val: float, frac_train:float,  target_dir: str, SEED=12345):
    index = femr.index.PatientIndex(dataset, num_proc=4)
    patient_ids = index.get_patient_ids()

    main_split = femr.splits.generate_hash_split(patient_ids, SEED, frac_test=frac_test)    
    train_val_split = femr.splits.generate_hash_split(main_split.train_patient_ids, SEED + 1, frac_test=frac_val / (frac_train + frac_val))
    train_ids = train_val_split.train_patient_ids
    val_ids = train_val_split.test_patient_ids
    test_ids = main_split.test_patient_ids
    print(f"Training set size: {len(train_ids)}")
    print(f"Validation set size: {len(val_ids)}")
    print(f"Test set size: {len(test_ids)}")

    split = PatientSplit(train_patient_ids=train_ids, test_patient_ids=test_ids, val_patient_ids=val_ids)
    split.save_to_csv(target_dir)


@dataclasses.dataclass
class PatientSplit:
    train_patient_ids: List[int]
    test_patient_ids: List[int]
    val_patient_ids: List[int]

    def save_to_csv(self, fname: str):
        with open(fname, "w") as f:
            writer = csv.DictWriter(f, ("patient_id", "split_name"))
            writer.writeheader()
            for train in self.train_patient_ids:
                writer.writerow({"patient_id": train, "split_name": "train"})
            for test in self.test_patient_ids:
                writer.writerow({"patient_id": test, "split_name": "test"})
            for val in self.val_patient_ids:
                writer.writerow({"patient_id": val, "split_name": "val"})

    @classmethod
    def load_from_csv(cls, fname: str):
        train_patient_ids: List[int] = []
        test_patient_ids: List[int] = []
        val_patient_ids: List[int] = []
        with open(fname, "r") as f:
            for row in csv.DictReader(f):
                if row["split_name"] == "train":
                    train_patient_ids.append(int(row["patient_id"]))
                elif row["split_name"] == "test":
                    test_patient_ids.append(int(row["patient_id"]))
                else:
                    val_patient_ids.append(int(row["patient_id"]))

        return PatientSplit(train_patient_ids=train_patient_ids, test_patient_ids=test_patient_ids, val_patient_ids=val_patient_ids)

    def split_dataset(self, dataset: Dataset, index: femr.index.PatientIndex) -> DatasetDict:
        train_indices = [index.get_index(patient_id) for patient_id in self.train_patient_ids]
        test_indices = [index.get_index(patient_id) for patient_id in self.test_patient_ids]
        val_indices = [index.get_index(patient_id) for patient_id in self.val_patient_ids]
        return DatasetDict(
            {
                "train": dataset.select(train_indices),
                "val": dataset.select(val_indices),
                "test": dataset.select(test_indices),
            }
        )
