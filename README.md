---
title: Dialogue Text Summarizer
emoji: 🧠
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
suggested_hardware: cpu-basic
---

# 🧠 Text Summarizer — Dialogue Summarization with T5

A fine-tuned **T5 (Text-to-Text Transfer Transformer)** model that summarizes multi-turn chat conversations, served through a **FastAPI** backend with a lightweight web UI.

Unlike most summarizers built for news articles or long documents, this project targets **conversational text** — the kind of unstructured, informal dialogue found in Slack threads, WhatsApp chats, or support tickets — using the [SAMSum](https://huggingface.co/datasets/samsum) dataset.

---

## ✨ Features

- Fine-tuned `t5-small` model specialized for dialogue-to-summary generation
- Custom text-cleaning pipeline (noise removal, whitespace/HTML normalization)
- Beam-search decoding (`num_beams=4`) for coherent, non-repetitive summaries
- REST API built with FastAPI (`/summarize/` endpoint)
- Simple browser-based UI for quick manual testing
- Saved model artifacts (safetensors) for fast, reproducible loading

---

## 🏗️ Project Architecture

```
Raw dialogue (chat text)
        │
        ▼
Text Cleaning (regex-based normalization)
        │
        ▼
T5 Tokenizer (max_length=512)
        │
        ▼
Fine-tuned T5 Model (beam search, num_beams=4)
        │
        ▼
Generated Summary
        │
        ▼
FastAPI (/summarize/) ──► Web UI (index.html)
```

---

## 📂 Project Structure

```
Text-Summarizer-Mini-Project/
├── app.py                      # FastAPI app: model loading, inference, API + UI routes
├── index.html                  # Front-end interface for the summarizer
├── text_summarizer.ipynb       # Data prep, fine-tuning, and evaluation notebook
├── saved_summary_model/        # Fine-tuned model + tokenizer (safetensors)
├── samsum-train.csv            # Training split (SAMSum dataset)
├── samsum-validation.csv       # Validation split
├── samsum-test.csv             # Test split
└── README.md
```

---

## 🧪 Dataset

**[SAMSum Corpus](https://huggingface.co/datasets/samsum)** — a dataset of messenger-style conversations paired with human-written summaries.

- Training subset: 4,000 samples (randomly sampled, `random_state=42`)
- Validation subset: 500 samples
- Fields used: `dialogue`, `summary`

---

## ⚙️ Model & Training

| Parameter | Value |
|---|---|
| Base model | `t5-small` |
| Framework | Hugging Face `transformers` (`Trainer` API) |
| Epochs | 6 |
| Batch size | 8 (train & eval) |
| Warmup steps | 500 |
| Weight decay | 0.01 |
| Max input length | 512 tokens |
| Max target length | 200 tokens |
| Decoding strategy | Beam search (`num_beams=4`, early stopping) |
| Final training loss | 0.82 |

**Preprocessing** included lowercasing, whitespace normalization, HTML-tag stripping, and line-break cleanup — applied identically at training and inference time (`clean_data()` in `app.py`) to avoid train/serve skew.

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. Install dependencies
```bash
pip install fastapi uvicorn transformers torch jinja2
```

### 3. Run the app
```bash
uvicorn app:app --reload
```

### 4. Open in browser
```
http://127.0.0.1:8000
```

### 5. Or call the API directly
```bash
curl -X POST "http://127.0.0.1:8000/summarize/" \
  -H "Content-Type: application/json" \
  -d '{"dialogue": "Alex: Are you coming to the meeting? Jordan: Yes, on my way now."}'
```

---

## 📌 Example

**Input dialogue:**
> Alex: Hey, have you noticed how many electric vehicles are on the road these days?
> Jordan: Yeah, I've definitely seen more. I'm thinking about buying one.
> Alex: What's making you consider an EV?
> Jordan: Mainly the lower running costs and less maintenance.

**Generated summary:**
> Jordan has seen more electric vehicles on the road lately and is considering buying one due to lower running costs.

---

## 🔭 Next Steps

- [ ] Evaluate with ROUGE-1 / ROUGE-2 / ROUGE-L metrics on the held-out test set
- [ ] Test generalization on longer, multi-topic conversations
- [ ] Experiment with `t5-base` for a quality/latency comparison
- [ ] Containerize with Docker for deployment
- [ ] Add batch summarization support to the API

---

## 🛠️ Tech Stack

`Python` · `PyTorch` · `Hugging Face Transformers` · `FastAPI` · `Jinja2` · `Pandas`

---

## 📄 License

This project is for educational/portfolio purposes.

---

## 🙋 Author

**Karthik**
B.Tech, Artificial Intelligence & Data Science
Open to Data Analyst / ML Engineer / AI Engineer roles
