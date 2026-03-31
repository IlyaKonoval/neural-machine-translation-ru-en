import time
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.inference import Translator

translator: Translator | None = None
config: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global translator, config

    config_path = Path("configs/config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = config["paths"]["best_model"]

    if not Path(checkpoint_path).exists():
        print(f"WARNING: Checkpoint not found at {checkpoint_path}")
        yield
        return

    translator = Translator.from_checkpoint(
        checkpoint_path=checkpoint_path,
        tokenizer_name=config["model"]["tokenizer"],
        device=device,
        embed_size=config["model"]["embed_size"],
        num_layers=config["model"]["num_layers"],
        heads=config["model"]["heads"],
        ff_hidden_size=config["model"]["ff_hidden_size"],
        dropout=config["model"]["dropout"],
        max_len=config["inference"]["max_len"],
    )
    print(f"Model loaded on {device}")
    yield


app = FastAPI(
    title="Transformer RU→EN Translation API",
    description="Neural machine translation from Russian to English using Transformer",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500, examples=["Привет, как дела?"])
    beam_size: int = Field(default=5, ge=1, le=10)


class TranslateResponse(BaseModel):
    source: str
    translation: str
    beam_size: int
    elapsed_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        model_loaded=translator is not None,
        device=str(translator.device) if translator else "n/a",
    )


@app.post("/translate", response_model=TranslateResponse)
async def translate(request: TranslateRequest):
    if translator is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Train the model first."
        )

    start = time.perf_counter()
    result = translator.translate(request.text, beam_size=request.beam_size)
    elapsed = (time.perf_counter() - start) * 1000

    return TranslateResponse(
        source=request.text,
        translation=result,
        beam_size=request.beam_size,
        elapsed_ms=round(elapsed, 2),
    )
