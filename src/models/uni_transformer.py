from __future__ import annotations

import collections
from typing import Any, Dict, List, Mapping, Optional

import datasets
import meds
import numpy as np
import torch
import torch.nn.functional as F
import transformers
import xformers.ops
from torch import nn
from tqdm import tqdm

import femr.models.config
import femr.models.processor
import femr.models.rmsnorm
import femr.models.tasks
import femr.models.xformers
from femr.models.transformer import (
    fixed_pos_embedding,
    apply_rotary_pos_emb,
    CLMBRTaskHead,
    LabeledPatientTaskHead,
)

from models.universal_processor import UniFEMRBatchProcessor
from models.univeral_tokenizer import UniversalTokenizer


class UniFEMREncoderLayer(nn.Module):
    def __init__(self, config: femr.models.config.FEMRTransformerConfig):
        super().__init__()
        self.config = config
        self.norm = femr.models.rmsnorm.RMSNorm(self.config.hidden_size)
        if self.config.hidden_act == "swiglu":
            hidden_mult = 2
        else:
            hidden_mult = 1

        self.input_proj = nn.Linear(
            self.config.hidden_size,
            self.config.hidden_size * 3 + hidden_mult * self.config.intermediate_size,
            bias=self.config.use_bias,
        )
        self.output_proj = nn.Linear(
            self.config.hidden_size + self.config.intermediate_size,
            self.config.hidden_size,
            bias=self.config.use_bias,
        )

    def forward(self, x, normed_ages, pos_embed, attn_bias):
        x = self.norm(x)

        if self.config.use_normed_ages:
            x[:, -2] = normed_ages.to(dtype=x.dtype)
            x[:, -1] = (normed_ages**2).to(dtype=x.dtype)

        transformed = self.input_proj(x)

        ff = transformed[:, : -self.config.hidden_size * 3]
        qkv = transformed[:, -self.config.hidden_size * 3 :]

        head_size = self.config.hidden_size // self.config.n_heads

        qkv = qkv.reshape(x.shape[0], 3, self.config.n_heads, head_size)

        q = apply_rotary_pos_emb(qkv[:, 0, :, :], pos_embed)
        k = apply_rotary_pos_emb(qkv[:, 1, :, :], pos_embed)
        v = qkv[:, 2, :, :]

        attn = femr.models.xformers.memory_efficient_attention_wrapper(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            attn_bias=attn_bias,
        )

        attn = attn.reshape(x.shape)

        if self.config.hidden_act == "gelu":
            ff = F.gelu(ff)
        elif self.config.hidden_act == "swiglu":
            x1, x2 = ff.chunk(2, dim=-1)
            ff = F.silu(x1) * x2

        combined = torch.concatenate((attn, ff), axis=-1)
        result = self.output_proj(combined)

        return result


class UniFEMRTransformer(nn.Module):
    def __init__(self, config: femr.models.config.FEMRTransformerConfig):
        super().__init__()
        self.config = config

        self.in_norm = femr.models.rmsnorm.RMSNorm(self.config.hidden_size)
        self.out_norm = femr.models.rmsnorm.RMSNorm(self.config.hidden_size)
        self.embed = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.layers = nn.ModuleList(
            [UniFEMREncoderLayer(config) for _ in range(self.config.n_layers)]
        )

        self.tinybert_model = transformers.AutoModel.from_pretrained(
            "huawei-noah/TinyBERT_General_4L_312D"
        )
        for param in self.tinybert_model.parameters():
            param.requires_grad = False

        self.cls_projection = nn.Linear(
            self.tinybert_model.config.hidden_size, self.config.hidden_size
        )

    def forward(self, batch):
        x = self.embed(batch["tokens"])
        x = self.in_norm(x)

        # Compute mask and filter non-zero tokens in a batch-wise manner
        non_zero_mask = batch["text_tokens"].sum(dim=-1) != 0
        if non_zero_mask.any():
            text_tokens_tensor = batch["text_tokens"][non_zero_mask].to(x.device)
            attention_mask = text_tokens_tensor != 0
            with torch.no_grad():
                tinybert_output = self.tinybert_model(
                    text_tokens_tensor, attention_mask
                )
                cls_embedding = tinybert_output.last_hidden_state[:, 0, :]
                projected_cls_embedding = self.cls_projection(cls_embedding)
                x[non_zero_mask] += projected_cls_embedding

        normed_ages = batch["normalized_ages"]
        pos_embed = fixed_pos_embedding(
            batch["ages"], self.config.hidden_size // self.config.n_heads, x.dtype
        )

        attn_bias = xformers.ops.fmha.attn_bias.BlockDiagonalMask.from_seqlens(
            batch["patient_lengths"].tolist()
        ).make_local_attention(self.config.attention_width)

        for layer in self.layers:
            x = x + layer(x, normed_ages, pos_embed, attn_bias)

        final = self.out_norm(x)

        return final


def remove_first_dimension(data: Any) -> Any:
    if isinstance(data, collections.abc.Mapping):
        return {k: remove_first_dimension(v) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        assert data.shape[0] == 1
        return data.squeeze(dim=0)
    elif isinstance(data, np.ndarray):
        assert data.shape[0] == 1
        return np.squeeze(data, axis=0)
    elif isinstance(data, (int, float, np.number, np.bool_)):
        return data
    else:
        raise RuntimeError("Could not convert item of type " + str(type(data)))


class UniFEMRModel(transformers.PreTrainedModel):
    config_class = femr.models.config.FEMRModelConfig

    def __init__(self, config: femr.models.config.FEMRModelConfig, **kwargs):
        # Allow the task config to be ovewritten
        if "task_config" in kwargs:
            config.task_config = kwargs["task_config"]

        super().__init__(config)

        self.transformer = UniFEMRTransformer(self.config.transformer_config)
        if self.config.task_config is not None:
            self.task_model = self.create_task_head()

    def create_task_head(self) -> nn.Module:
        hidden_size = self.config.transformer_config.hidden_size
        task_type = self.config.task_config.task_type
        task_kwargs = self.config.task_config.task_kwargs
        if task_type == "clmbr":
            return CLMBRTaskHead(hidden_size, **task_kwargs)
        elif task_type == "labeled_patients":
            return LabeledPatientTaskHead(hidden_size, **task_kwargs)
        else:
            raise ValueError(f"Unknown task type {task_type}")

    def forward(
        self,
        batch: Mapping[str, Any],
        return_loss=True,
        return_logits=False,
        return_reprs=False,
    ):
        # Need a return_loss parameter for transformers.Trainer to work properly
        assert return_loss

        batch = remove_first_dimension(batch)

        features = self.transformer(batch["transformer"])
        if "task" in batch and self.config.task_config is not None:
            features = features.reshape(-1, features.shape[-1])
            features = features[batch["transformer"]["label_indices"], :]
            loss, result = self.task_model(
                features, batch["task"], return_logits=return_logits
            )
            if return_reprs:
                result["representations"] = features
            if return_logits or return_reprs:
                result["timestamps"] = batch["transformer"]["timestamps"][
                    batch["transformer"]["label_indices"]
                ]
                result["patient_ids"] = batch["patient_ids"][
                    batch["transformer"]["label_indices"]
                ]
            return loss, result
        else:
            loss = 0
            features = features.reshape(-1, features.shape[-1])
            if "task" in batch:
                features = features[batch["transformer"]["label_indices"], :]
                result = {
                    "timestamps": batch["transformer"]["timestamps"][
                        batch["transformer"]["label_indices"]
                    ],
                    "patient_ids": batch["patient_ids"][
                        batch["transformer"]["label_indices"]
                    ],
                    "representations": features,
                }
            else:
                result = {
                    "timestamps": batch["transformer"]["timestamps"],
                    "patient_ids": batch["patient_ids"],
                    "representations": features,
                }

            return loss, result


def compute_features(
    dataset: datasets.Dataset,
    model_path: str,
    tokenizer: UniversalTokenizer,
    labels: List[meds.Label],
    num_proc: int = 1,
    tokens_per_batch: int = 1024,
    device: Optional[torch.device] = None,
) -> Dict[str, np.ndarray]:
    task = femr.models.tasks.LabeledPatientTask(labels)

    index = femr.index.PatientIndex(dataset, num_proc=num_proc)

    model = UniFEMRModel.from_pretrained(model_path, task_config=task.get_task_config())
    # tokenizer = UniversalTokenizer.from_pretrained(model_path)
    processor = UniFEMRBatchProcessor(tokenizer, task=task)

    filtered_data = task.filter_dataset(dataset, index)

    if device:
        model = model.to(device)

    batches = processor.convert_dataset(
        filtered_data,
        tokens_per_batch=tokens_per_batch,
        min_patients_per_batch=1,
        num_proc=num_proc,
    )

    batches.set_format("pt", device=device)

    all_patient_ids = []
    all_feature_times = []
    all_representations = []

    for batch in tqdm(batches, desc="Processing batches"):
        batch = processor.collate([batch])["batch"]
        with torch.no_grad():
            _, result = model(batch, return_reprs=True)
            all_patient_ids.append(result["patient_ids"].cpu().numpy())
            all_feature_times.append(result["timestamps"].cpu().numpy())
            all_representations.append(result["representations"].cpu().numpy())

    return {
        "patient_ids": np.concatenate(all_patient_ids),
        "feature_times": np.concatenate(all_feature_times).astype("datetime64[s]"),
        "features": np.concatenate(all_representations),
    }
