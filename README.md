

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
- [x] Add Docker and Render deployment configuration
- [ ] Add batch summarization support to the API

---

## Deploy on Render

This repository includes [`render.yaml`](render.yaml), which creates a Render
web service with the correct build command, start command, Python version, and
health check.

1. Push this project to GitHub. Keep `saved_summary_model/` and the SAMSum CSV
   files out of Git; they are intentionally ignored because of their size.
   Upload the model folder to the Hugging Face model repository named by
   `MODEL_PATH` before deploying. This project's default is
   `Karthiktelukutla2004/text-summarizer`.
2. In the [Render Dashboard](https://dashboard.render.com/), choose **New +**
   > **Blueprint**, connect GitHub, and select the repository.
3. Review the proposed `text-summarizer` service and click **Apply**. Render
   reads the configuration from `render.yaml`.
4. Wait for the first deployment to complete. The app downloads
   `Karthiktelukutla2004/text-summarizer` from Hugging Face on first start,
   which can make that first start slower.
5. Open the generated `https://<service-name>.onrender.com` URL. Confirm
   `https://<service-name>.onrender.com/health` returns `{"status":"ok"}`.

For a manual Render setup, create a **Python Web Service** with:

| Setting | Value |
|---|---|
| Build command | `pip install --upgrade pip && pip install -r requirements.txt` |
| Start command | `uvicorn app:app --host 0.0.0.0 --port $PORT` |
| Health check path | `/health` |
| Environment variables | `PYTHON_VERSION=3.11.9`, `MODEL_PATH=Karthiktelukutla2004/text-summarizer` |

The Blueprint uses Render's free plan. It can spin down after inactivity, so
the next request may take longer while the service wakes and loads the model.
Use a paid instance (and optionally a persistent disk for the model cache) for
steadier production response times.

### Private Hugging Face model

If the model repository is private, create a Hugging Face access token with
**Read** permission. In Render, open the service's **Environment** page, add an
environment variable named `HF_TOKEN`, paste the token as its value, save it,
and redeploy. Do not add this token to `render.yaml`, GitHub, or source code.

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
