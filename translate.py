import argparse

import yaml
import torch

from src.inference import Translator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--beam-size", type=int, default=None)
    parser.add_argument("--text", default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    checkpoint_path = args.checkpoint or cfg["paths"]["best_model"]
    beam_size = args.beam_size if args.beam_size is not None else cfg["inference"]["beam_size"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading model from {checkpoint_path}...")

    translator = Translator.from_checkpoint(
        checkpoint_path=checkpoint_path,
        tokenizer_name=cfg["model"]["tokenizer"],
        device=device,
        embed_size=cfg["model"]["embed_size"],
        num_layers=cfg["model"]["num_layers"],
        heads=cfg["model"]["heads"],
        ff_hidden_size=cfg["model"]["ff_hidden_size"],
        dropout=cfg["model"]["dropout"],
        max_len=cfg["inference"]["max_len"],
    )
    print("Model loaded.\n")

    if args.text:
        result = translator.translate(args.text, beam_size=beam_size)
        print(f"RU: {args.text}")
        print(f"EN: {result}")
    else:
        print("Interactive mode. Type 'quit' to exit.\n")
        while True:
            try:
                text = input("RU > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if text.lower() in ("quit", "exit", "q"):
                break
            if not text:
                continue
            result = translator.translate(text, beam_size=beam_size)
            print(f"EN > {result}\n")


if __name__ == "__main__":
    main()
