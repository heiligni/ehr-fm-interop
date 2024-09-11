import os
import sys

# Adding the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(os.path.join(parent_dir, "src"))

import xgboost
from transformers import TrainingArguments, Trainer, TrainerCallback
import matplotlib.pyplot as plt

from utility.logs import log

if __name__ == "__main__":
    log("Test script")
    print("This is a test scipt to test the execution environment")
