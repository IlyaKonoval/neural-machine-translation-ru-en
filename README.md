# Transformer RU→EN Machine Translation

Production-ready нейронный машинный перевод с русского на английский на основе архитектуры Transformer, написанной с нуля на PyTorch.

## Архитектура

```
Russian text → [BERT Tokenizer] → [Transformer Encoder] → [Cross-Attention] → [Transformer Decoder] → [BERT Tokenizer] → English text
```

| Компонент | Описание |
|-----------|----------|
| Encoder | 4 слоя, Multi-Head Self-Attention (8 голов) + FFN |
| Decoder | 4 слоя, Masked Self-Attention + Cross-Attention + FFN |
| Positional Encoding | Синусоидальное позиционное кодирование |
| Tokenizer | BERT (`bert-base-uncased`) |
| Decoding | Greedy / Beam Search |

## Структура проекта

```
├── src/
│   ├── data/           # Preprocessing, Dataset, DataLoader
│   ├── model/          # Transformer (from scratch)
│   ├── training/       # Train/evaluate, checkpointing
│   └── inference/      # Translator с greedy и beam search
├── api/
│   └── app.py          # FastAPI REST API
├── frontend/
│   └── streamlit_app.py
├── tests/
├── configs/
│   └── config.yaml
├── notebooks/          # Оригинальные notebooks
├── train.py
├── translate.py
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/  # CI/CD
```

## Быстрый старт

### Установка

```bash
git clone https://github.com/yourusername/seq2seq-LSTM-en-ru.git
cd seq2seq-LSTM-en-ru
pip install -r requirements.txt
```

### Обучение

Скачайте [датасет](https://www.kaggle.com/datasets) `rus.txt` и поместите в `data/rus.txt`.

```bash
python train.py --config configs/config.yaml --data data/rus.txt
```

### Перевод (CLI)

```bash
python translate.py --text "Привет, как дела?"
python translate.py  # интерактивный режим
```

### REST API

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Swagger: http://localhost:8000/docs

```bash
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Привет мир", "beam_size": 5}'
```

### Streamlit UI

```bash
streamlit run frontend/streamlit_app.py
```

### Docker

```bash
docker-compose up --build
```

- API: http://localhost:8000
- Frontend: http://localhost:8501

## Тестирование

```bash
pytest tests/ -v
```

## CI/CD

GitHub Actions: lint (ruff) + pytest на каждый push/PR в `main`.

## Конфигурация

| Параметр | Значение |
|----------|----------|
| Embedding dim | 256 |
| Attention heads | 8 |
| Encoder/Decoder layers | 4 |
| FFN hidden size | 1024 |
| Dropout | 0.1 |
| Optimizer | Adam (lr=0.0001) |
| LR Scheduler | ReduceLROnPlateau |
| Batch size | 32 |
| Max sequence length | 128 |
| Tokenizer | bert-base-uncased |

## Датасет

[Tatoeba](https://www.manythings.org/anki/) — 363,386 параллельных пар RU-EN в формате TSV.

## Технологии

- **ML:** PyTorch, Transformers (tokenizer), NLTK
- **Backend:** FastAPI, Uvicorn
- **Frontend:** Streamlit
- **DevOps:** Docker, GitHub Actions
- **Testing:** Pytest
- **Linting:** Ruff

## Лицензия

MIT License — см. [LICENSE](LICENSE)
