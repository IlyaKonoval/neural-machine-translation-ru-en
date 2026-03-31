from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoTokenizer

from ..data.preprocessing import clean_text
from ..model import Transformer


class Translator:
    def __init__(
        self,
        model: Transformer,
        tokenizer: AutoTokenizer,
        device: torch.device,
        max_len: int = 128,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_len = max_len
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        tokenizer_name: str = "bert-base-uncased",
        device: torch.device | str = "cpu",
        embed_size: int = 256,
        num_layers: int = 4,
        heads: int = 8,
        ff_hidden_size: int = 1024,
        dropout: float = 0.1,
        max_len: int = 128,
    ) -> "Translator":
        device = torch.device(device) if isinstance(device, str) else device
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, clean_up_tokenization_spaces=True)

        vocab_size = len(tokenizer)
        pad_idx = tokenizer.pad_token_id

        model = Transformer(
            src_vocab_size=vocab_size,
            tgt_vocab_size=vocab_size,
            src_pad_idx=pad_idx,
            tgt_pad_idx=pad_idx,
            embed_size=embed_size,
            num_layers=num_layers,
            heads=heads,
            ff_hidden_size=ff_hidden_size,
            dropout=dropout,
        ).to(device)

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        return cls(model, tokenizer, device, max_len)

    def translate(self, sentence: str, beam_size: int = 1) -> str:
        cleaned = clean_text(sentence)
        tokens = self.tokenizer(cleaned, return_tensors="pt").input_ids.to(self.device)

        if beam_size <= 1:
            output_ids = self._greedy_decode(tokens)
        else:
            output_ids = self._beam_search(tokens, beam_size)

        return self.tokenizer.decode(output_ids, skip_special_tokens=True)

    def _greedy_decode(self, src: torch.Tensor) -> list[int]:
        target = torch.tensor([[self.tokenizer.cls_token_id]], device=self.device)

        with torch.no_grad():
            for _ in range(self.max_len):
                output = self.model(src, target)
                pred_token = output.argmax(2)[:, -1].item()
                target = torch.cat([target, torch.tensor([[pred_token]], device=self.device)], dim=1)
                if pred_token == self.tokenizer.sep_token_id:
                    break

        return target.squeeze(0).tolist()

    def _beam_search(self, src: torch.Tensor, beam_size: int) -> list[int]:
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id

        beams = [(0.0, [cls_id])]
        completed = []

        with torch.no_grad():
            for _ in range(self.max_len):
                candidates = []
                for log_prob, seq in beams:
                    if seq[-1] == sep_id:
                        completed.append((log_prob, seq))
                        continue

                    tgt_tensor = torch.tensor([seq], device=self.device)
                    output = self.model(src, tgt_tensor)
                    logits = output[:, -1, :]
                    log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)
                    topk_probs, topk_ids = log_probs.topk(beam_size)

                    for i in range(beam_size):
                        token_id = topk_ids[i].item()
                        new_log_prob = log_prob + topk_probs[i].item()
                        candidates.append((new_log_prob, seq + [token_id]))

                if not candidates:
                    break

                candidates.sort(key=lambda x: x[0] / len(x[1]), reverse=True)
                beams = candidates[:beam_size]

        for log_prob, seq in beams:
            completed.append((log_prob, seq))

        if not completed:
            return [cls_id]

        best = max(completed, key=lambda x: x[0] / len(x[1]))
        return best[1]
