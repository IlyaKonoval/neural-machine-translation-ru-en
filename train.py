import argparse
from pathlib import Path

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer

from src.data import load_data, create_dataloaders
from src.model import Transformer
from src.training import train_epoch, evaluate_epoch, save_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--data", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_path = args.data or cfg["data"]["data_path"]
    n_epochs = args.epochs or cfg["training"]["n_epochs"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["tokenizer"], clean_up_tokenization_spaces=True)
    print(f"Tokenizer: {cfg['model']['tokenizer']} ({len(tokenizer)} tokens)")

    print("Loading data...")
    train_data, val_data, test_data = load_data(data_path, max_samples=cfg["data"]["max_samples"])
    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data, tokenizer,
        batch_size=cfg["training"]["batch_size"],
        max_length=cfg["data"]["max_length"],
        num_workers=cfg["training"]["num_workers"],
    )

    vocab_size = len(tokenizer)
    pad_idx = tokenizer.pad_token_id

    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        src_pad_idx=pad_idx,
        tgt_pad_idx=pad_idx,
        embed_size=cfg["model"]["embed_size"],
        num_layers=cfg["model"]["num_layers"],
        heads=cfg["model"]["heads"],
        ff_hidden_size=cfg["model"]["ff_hidden_size"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=cfg["training"]["scheduler"]["patience"],
        factor=cfg["training"]["scheduler"]["factor"],
        min_lr=cfg["training"]["scheduler"]["min_lr"],
    )
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    best_val_loss = float("inf")
    print(f"\nTraining for {n_epochs} epochs...")
    print("-" * 60)

    for epoch in range(1, n_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, cfg["training"]["clip"], device)
        val_loss = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:02d}/{n_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")

        save_checkpoint(
            model, optimizer, epoch, train_loss, val_loss,
            Path(cfg["paths"]["checkpoint_dir"]) / f"checkpoint_epoch_{epoch:02d}.pt",
            config=cfg,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss,
                cfg["paths"]["best_model"],
                config=cfg,
            )
            print(f"  -> Best model saved (val_loss: {val_loss:.4f})")

    print("-" * 60)
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
