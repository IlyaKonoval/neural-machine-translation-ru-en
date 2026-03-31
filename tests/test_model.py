import torch
import pytest
from src.model import Transformer


@pytest.fixture
def params():
    return {
        "vocab_size": 100,
        "pad_idx": 0,
        "embed_size": 32,
        "num_layers": 2,
        "heads": 4,
        "ff_hidden_size": 64,
        "dropout": 0.1,
        "batch_size": 4,
        "src_len": 10,
        "tgt_len": 8,
    }


@pytest.fixture
def model(params):
    return Transformer(
        src_vocab_size=params["vocab_size"],
        tgt_vocab_size=params["vocab_size"],
        src_pad_idx=params["pad_idx"],
        tgt_pad_idx=params["pad_idx"],
        embed_size=params["embed_size"],
        num_layers=params["num_layers"],
        heads=params["heads"],
        ff_hidden_size=params["ff_hidden_size"],
        dropout=params["dropout"],
    )


class TestTransformer:
    def test_forward_output_shape(self, model, params):
        src = torch.randint(
            1, params["vocab_size"], (params["batch_size"], params["src_len"])
        )
        tgt = torch.randint(
            1, params["vocab_size"], (params["batch_size"], params["tgt_len"])
        )
        output = model(src, tgt)
        assert output.shape == (
            params["batch_size"],
            params["tgt_len"],
            params["vocab_size"],
        )

    def test_src_mask(self, model, params):
        src = torch.tensor([[1, 2, 3, 0, 0]])
        mask = model.make_src_mask(src)
        assert mask.shape == (1, 1, 1, 5)
        assert mask[0, 0, 0, 0].item() is True
        assert mask[0, 0, 0, 3].item() is False

    def test_tgt_mask_causal(self, model, params):
        tgt = torch.tensor([[1, 2, 3]])
        mask = model.make_tgt_mask(tgt)
        assert mask.shape == (1, 1, 3, 3)
        assert mask[0, 0, 0, 1].item() is False
        assert mask[0, 0, 1, 0].item() is True

    def test_inference_mode(self, model, params):
        model.eval()
        src = torch.randint(1, params["vocab_size"], (1, params["src_len"]))
        tgt = torch.randint(1, params["vocab_size"], (1, 1))
        with torch.no_grad():
            output = model(src, tgt)
        assert output.shape == (1, 1, params["vocab_size"])
