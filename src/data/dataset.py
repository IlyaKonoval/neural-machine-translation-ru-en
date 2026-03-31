from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import Dataset as HFDataset

from .preprocessing import clean_text


def load_data(
    data_path: str | Path,
    max_samples: int | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(data_path, delimiter="\t", header=None)
    df.rename(columns={0: "en", 1: "ru", 2: "comment"}, inplace=True)
    df = df[["en", "ru"]]

    if max_samples is not None:
        df = df.head(max_samples)

    df["en"] = df["en"].apply(clean_text)
    df["ru"] = df["ru"].apply(clean_text)

    train_data, temp_data = train_test_split(df, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    return train_data, val_data, test_data


def _tokenize_fn(examples, tokenizer, max_length=128):
    source = tokenizer(examples["ru"], padding="max_length", truncation=True, max_length=max_length)
    target = tokenizer(examples["en"], padding="max_length", truncation=True, max_length=max_length)
    source["labels"] = target["input_ids"]
    return source


def _collate_fn(batch, pad_token_id):
    input_ids = [torch.tensor(item["input_ids"]) for item in batch]
    labels = [torch.tensor(item["labels"]) for item in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=pad_token_id)

    return {"input_ids": input_ids, "labels": labels}


def create_dataloaders(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
    tokenizer: AutoTokenizer,
    batch_size: int = 32,
    max_length: int = 128,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = HFDataset.from_pandas(train_data)
    val_ds = HFDataset.from_pandas(val_data)
    test_ds = HFDataset.from_pandas(test_data)

    tokenize = lambda examples: _tokenize_fn(examples, tokenizer, max_length)
    train_ds = train_ds.map(tokenize, batched=False)
    val_ds = val_ds.map(tokenize, batched=False)
    test_ds = test_ds.map(tokenize, batched=False)

    collate = lambda batch: _collate_fn(batch, tokenizer.pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=num_workers)

    return train_loader, val_loader, test_loader
