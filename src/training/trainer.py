import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_epoch(
    model: nn.Module,
    iterator: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    clip: float,
    device: torch.device,
) -> float:
    model.train()
    epoch_loss = 0.0

    progress = tqdm(iterator, desc="Training", leave=False)
    for batch in progress:
        src = batch["input_ids"].to(device)
        tgt = batch["labels"].to(device)

        optimizer.zero_grad()
        output = model(src, tgt[:, :-1])
        output = output.contiguous().view(-1, output.shape[-1])
        tgt_flat = tgt[:, 1:].contiguous().view(-1)

        loss = criterion(output, tgt_flat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.3f}")

    return epoch_loss / len(iterator)


def evaluate_epoch(
    model: nn.Module,
    iterator: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    epoch_loss = 0.0

    progress = tqdm(iterator, desc="Evaluating", leave=False)
    with torch.no_grad():
        for batch in progress:
            src = batch["input_ids"].to(device)
            tgt = batch["labels"].to(device)

            output = model(src, tgt[:, :-1])
            output = output.contiguous().view(-1, output.shape[-1])
            tgt_flat = tgt[:, 1:].contiguous().view(-1)

            loss = criterion(output, tgt_flat)
            epoch_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.3f}")

    return epoch_loss / len(iterator)
