import os

from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch
import re
from pathlib import Path
from fastapi.templating import Jinja2Templates  # UI
from fastapi.responses import HTMLResponse

app = FastAPI(title="Text Summarizer App", description="Text Summarization using T5", version="1.0")

# Prefer the bundled model so local development works without a connection.
# If it is absent (as in the deployed repository), use the published model.
# MODEL_PATH may point to another local directory or a Hugging Face model ID.
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_DIR / "saved_summary_model"
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    str(DEFAULT_MODEL_PATH)
    if DEFAULT_MODEL_PATH.is_dir()
    else "karthik2004-tech/t5-dialogue-summarizer",
)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)

# device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)
model.eval()

templates = Jinja2Templates(directory=str(PROJECT_DIR))

class DialogueInput(BaseModel):
    dialogue: str

def clean_data(text):
    text = re.sub(r"\r\n", " ", text)  # lines
    text = re.sub(r"\s+", " ", text)  # spaces
    text = re.sub(r"<.*?>", " ", text)  # html tags <p> <h1>
    text = text.strip().lower()
    return text

def summarize_dialogue(dialogue: str) -> str:
    dialogue = clean_data(dialogue)  # clean

    if not dialogue:
        return ""

    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.inference_mode():
        targets = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            num_beams=4,
            early_stopping=True
        )

    # decoded our output
    summary = tokenizer.decode(targets[0], skip_special_tokens=True)  # EOS, SEP
    return summary


# API endpoints
@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}
