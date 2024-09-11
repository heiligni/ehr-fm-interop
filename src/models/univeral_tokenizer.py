from __future__ import annotations

import collections
import datetime
import functools
import math
import os
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import meds
import msgpack
import transformers
from transformers import BertTokenizer
import numpy as np

import femr.hf_utils
import femr.stat_utils


def train_tokenizer(
    dataset,
    vocab_size: int,
    num_proc: int = 1,
) -> UniversalTokenizer:
    """Train a FEMR tokenizer from the given dataset"""
    statistics = femr.hf_utils.aggregate_over_dataset(
        dataset,
        functools.partial(
            map_statistics,
            num_patients=len(dataset),
        ),
        agg_statistics,
        num_proc=num_proc,
        batch_size=1_000,
    )
    return UniversalTokenizer(
        convert_statistics_to_msgpack(statistics, vocab_size),
    )


def agg_statistics(stats1, stats2):
    stats1["age_stats"].combine(stats2["age_stats"])

    for k, v in stats2["code_counts"].items():
        stats1["code_counts"][k] += v

    return stats1


def normalize_unit(unit):
    if unit:
        return unit.lower().replace(" ", "")
    else:
        return None


def map_statistics(
    batch,
    *,
    num_patients: int,
    frac_values=0.05,
) -> Mapping[str, Any]:
    age_stats = femr.stat_utils.OnlineStatistics()
    code_counts: Dict[str, float] = collections.defaultdict(float)

    for events in batch["events"]:
        total_events = 0
        for event in events:
            for measurement in event["measurements"]:
                total_events += 1

        if total_events == 0:
            continue

        weight = 1.0 / (num_patients * total_events)
        birth_date = events[0]["time"]
        for event in events:
            for measurement in event["measurements"]:
                if event["time"] != birth_date:
                    age_stats.add(weight, (event["time"] - birth_date).total_seconds())
                code_counts[measurement["code"]] += weight

    return {
        "age_stats": age_stats,
        "code_counts": code_counts,
    }


def convert_statistics_to_msgpack(
    statistics,
    vocab_size: int,
):
    vocab = []

    for code, weight in statistics["code_counts"].items():
        entry = {
            "type": "code",
            "code_string": code,
            "weight": weight * math.log(weight) + (1 - weight) * math.log(1 - weight),
        }
        vocab.append(entry)

    vocab.sort(key=lambda a: a["weight"])
    vocab = vocab[:vocab_size]

    result = {
        "vocab": vocab,
        "is_hierarchical": False,
        "age_stats": {
            "mean": statistics["age_stats"].mean(),
            "std": statistics["age_stats"].standard_deviation(),
        },
    }

    return result


class UniversalTokenizer(transformers.utils.PushToHubMixin):
    def __init__(self, dictionary: Mapping[str, Any]):
        self.dictionary = dictionary

        self.dictionary = dictionary
        vocab = dictionary["vocab"]

        self.code_lookup = {}

        self.vocab_size = len(vocab)

        for i, dict_entry in enumerate(vocab):
            if dict_entry["type"] == "code":
                self.code_lookup[dict_entry["code_string"]] = i

        self.text_tokenizer = BertTokenizer.from_pretrained(
            "huawei-noah/TinyBERT_General_4L_312D"
        )

    @classmethod
    def from_pretrained(
        self,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        **kwargs,
    ):
        """
        Load the FEMR tokenizer.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:
                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                      user or organization name, like `dbmdz/bert-base-german-cased`.
                    - A path to a *directory* containing tokenization data saved using
                      [`save_pretrained`], e.g., `./my_data_directory/`.
            ontology: An ontology object for hierarchical tokenizers
            kwargs: Arguments for loading to pass to transformers.utils.hub.cached_file

        Returns:
            A FEMR Tokenizer
        """

        dictionary_file = transformers.utils.hub.cached_file(
            pretrained_model_name_or_path, "dictionary.msgpack", **kwargs
        )

        with open(dictionary_file, "rb") as f:
            dictionary = msgpack.load(f)

        return UniversalTokenizer(dictionary)

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        push_to_hub: bool = False,
        **kwargs,
    ):
        """
        Save the FEMR tokenizer.


        This method make sure the batch processor can then be re-loaded using the
        .from_pretrained class method.

        Args:
            save_directory (`str` or `os.PathLike`): The path to a directory where the tokenizer will be saved.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`Dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`PushToHubMixin.push_to_hub`] method.
        """
        assert not os.path.isfile(
            save_directory
        ), f"Provided path ({save_directory}) should be a directory, not a file"

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", str(save_directory).split(os.path.sep)[-1])
            repo_id = self._create_repo(repo_id, **kwargs)
            files_timestamps = self._get_files_timestamps(save_directory)

        with open(os.path.join(save_directory, "dictionary.msgpack"), "wb") as f:
            msgpack.dump(self.dictionary, f)

        if push_to_hub:
            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=kwargs.get("token"),
            )

    def start_patient(self):
        """Compute per-patient statistics that are required to generate features."""

        # This is currently a null-op, but is required for cost featurization
        pass

    def get_feature_codes(
        self, _time: datetime.datetime, measurement: meds.Measurement
    ) -> Tuple[List[int], Optional[List[float]]]:
        """Get codes for the provided measurement and time"""
        value = self.code_lookup.get(measurement["code"])
        text_tokens = None

        max_length = 512

        if "text_value" in measurement:
            text = measurement["text_value"]
            if text:
                text_tokens = self.text_tokenizer.encode(
                    text,
                    add_special_tokens=True,
                    max_length=max_length,
                    truncation=True,
                    padding="max_length",
                )

        # If no text tokens are found, create an array of zeros with the same length
        if text_tokens is None:
            text_tokens = np.zeros(max_length, dtype=np.int32)
        else:
            text_tokens = np.array(text_tokens, dtype=np.int32)

        if value is not None:
            return [value], text_tokens
        else:
            return [], None

    def normalize_age(self, age: datetime.timedelta) -> float:
        return (age.total_seconds() - self.dictionary["age_stats"]["mean"]) / (
            self.dictionary["age_stats"]["std"]
        )
