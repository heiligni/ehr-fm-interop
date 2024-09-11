import sys
import os


# Adding the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(os.path.join(parent_dir, "src"))

import argparse
from femr.models.tokenizer import FEMRTokenizer
from femr.models.tasks import CLMBRTask
from femr.models.config import FEMRTransformerConfig, FEMRModelConfig
import torch
from transformers import TrainingArguments, Trainer, TrainerCallback
from transformers.trainer_callback import (
    TrainerControl,
    TrainerState,
    EarlyStoppingCallback,
)
import json
from datetime import datetime

from models.processor import FEMRBatchProcessor
from models.model_names import MODEL_NAMES
from models.transformer import FEMRModel
from models.uni_transformer import UniFEMRModel
from models.universal_processor import UniFEMRBatchProcessor
from utility.data import load_dataset
from utility.logs import log_time


class LossLoggerCallback(TrainerCallback):
    def __init__(self, output_dir) -> None:
        self.train_losses = []
        self.eval_losses = []
        self.output_dir = output_dir

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs,
    ):
        if logs is not None:
            # Use "loss" key for train_loss
            if "train_loss" in logs:
                self.train_losses.append(logs["train_loss"])
                print(f"Logged train loss: {logs['train_loss']}")
            else:
                print("No 'loss' key found in logs.")

            if "eval_loss" in logs:
                self.eval_losses.append(logs["eval_loss"])
                print(f"Logged eval loss: {logs['eval_loss']}")
            else:
                print("No 'eval_loss' key found in logs.")

            # Save the losses periodically
            # if state.global_step % 100 == 0:
            print("Saving losses...")
            self.save_losses()

    def save_losses(self):
        loss_data = {"train_losses": self.train_losses, "eval_losses": self.eval_losses}
        with open(os.path.join(self.output_dir, "losses.json"), "w") as f:
            json.dump(loss_data, f)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self.save_losses()

        final_results = {
            "learning_rate": args.learning_rate,
            "eval_loss": min(self.eval_losses) if self.eval_losses else None,
        }

        # Save to results.json in text mode 'w'
        with open(os.path.join(self.output_dir, "results.json"), "w") as f:
            json.dump(final_results, f)


def pretrain_clmbr_model(train_batches, args):
    model_path = os.path.join(output_path, "fm")

    if args.model_name == "CLMBR-T-base":
        print(f"Loading default tokenizer from {model_path}")
        tokenizer = FEMRTokenizer.from_pretrained(model_path)
    else:
        print(f"Loading lab tokenizer from {model_path}")
        tokenizer = FEMRTokenizer.from_pretrained(model_path)

    print(f"Initiating pretraining with the following parameters: {args}")
    transformer_config = FEMRTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        is_hierarchical=tokenizer.is_hierarchical,  # False for default tokenizer
        n_layers=args.n_layers,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        n_heads=args.n_heads,
    )

    clmbr_task = CLMBRTask(clmbr_vocab_size=8192)

    if args.model_name == "CLMBR-T-base":
        processor = FEMRBatchProcessor(tokenizer, clmbr_task)

    elif args.model_name == "CLMBR-T-lab":
        processor = UniFEMRBatchProcessor(tokenizer, clmbr_task)
    else:
        raise ValueError(f"Unknown model name: {args.model_name}")
    config = FEMRModelConfig.from_transformer_task_configs(
        transformer_config, clmbr_task.get_task_config()
    )

    if args.model_name == "CLMBR-T-base":
        model = FEMRModel(config)
    else:
        model = UniFEMRModel(config)

    lr_str = f"lr_{args.learning_rate:.0e}".replace("-", "m")

    trainer_config = TrainingArguments(
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        output_dir=os.path.join(model_path, lr_str, "tmp_trainer"),
        remove_unused_columns=False,
        num_train_epochs=10,
        eval_strategy="epoch",
        logging_strategy="epoch",
        prediction_loss_only=True,
        save_strategy="epoch",
        save_total_limit=2,
        greater_is_better=False,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        learning_rate=args.learning_rate,
    )

    loss_logger = LossLoggerCallback(os.path.join(model_path, lr_str))

    early_stopping = EarlyStoppingCallback(early_stopping_patience=1)

    trainer = Trainer(
        model=model,
        data_collator=processor.collate,
        train_dataset=train_batches["train"],
        eval_dataset=train_batches["val"],
        args=trainer_config,
        callbacks=[loss_logger, early_stopping],
    )

    # Check if the checkpoint directory exists and contains checkpoints
    last_checkpoint = None
    if os.path.isdir(trainer_config.output_dir):
        checkpoints = [
            os.path.join(trainer_config.output_dir, d)
            for d in os.listdir(trainer_config.output_dir)
            if os.path.isdir(os.path.join(trainer_config.output_dir, d))
            and "checkpoint" in d
        ]
        if checkpoints:
            last_checkpoint = max(checkpoints, key=os.path.getmtime)
            print(f"Resuming from checkpoint: {last_checkpoint}")
        else:
            print("No checkpoints found. Starting from scratch.")

    before_training = datetime.now()
    trainer.train(resume_from_checkpoint=last_checkpoint)
    log_time(before_training, datetime.now(), "training")

    model.save_pretrained(os.path.join(model_path, lr_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--output-path", type=str)
    parser.add_argument("--n-layers", type=int)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--intermediate-size", type=int)
    parser.add_argument("--n-heads", type=int)
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--num-procs", type=int, default=4)
    parser.add_argument(
        "--model-name", type=str, default="CLMBR-T-base", choices=MODEL_NAMES
    )
    parser.add_argument("--learning-rate", type=float)

    args = parser.parse_args()

    output_path = args.output_path
    vocab_size = args.vocab_size
    num_proc = args.num_procs

    print(f"Detected GPUs: {torch.cuda.device_count()}")

    model_path = os.path.join(output_path, "fm")

    train_batches = load_dataset(os.path.join(output_path, "train_batches"))

    train_batches.set_format("pt")

    # Continue with the rest of the code
    pretrain_clmbr_model(train_batches, args)
