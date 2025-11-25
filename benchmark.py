import os
import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer
from model import TinyGPT  # your classifier


MODEL_DIR = './tiny_gpt_runs'
MODEL_FILE = 'model_epoch3.pt'
TOKENIZER_NAME = 'gpt2'

MAX_LEN = 64
D_MODEL = 128
N_LAYERS = 4
N_HEADS = 4
NUM_CLASSES = 2
LABELS = {0: "Negative", 1: "Positive"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Load Tokenizer
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

# -------------------------
# Initialize Model
# -------------------------
vocab_size = tokenizer.vocab_size
model = TinyGPT(
    vocab_size=vocab_size,
    max_len=MAX_LEN,
    d_model=D_MODEL,
    n_layers=N_LAYERS,
    n_heads=N_HEADS,
    num_classes=NUM_CLASSES
).to(device)

# -------------------------
# Load Model Weights
# -------------------------
model_path = os.path.join(MODEL_DIR, MODEL_FILE)
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model weights not found at {model_path}")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f"Loaded model weights from {MODEL_FILE}")

# -------------------------
# Helper: Encode sentences
# -------------------------
def _encode(texts):
    enc = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=MAX_LEN,
        return_tensors='pt'
    )
    return enc['input_ids'].to(device), enc['attention_mask'].to(device)

# -------------------------
# Classification function
# -------------------------
def classify(sentence):
    ids, mask = _encode([sentence])
    with torch.no_grad():
        logits = model(ids, attention_mask=mask)
        probs = torch.softmax(logits, dim=-1)[0]
    pred = probs.argmax().item()
    return pred

# -------------------------
# Load SST-2 Dataset
# -------------------------
dataset = load_dataset("gimmaru/glue-sst2")
split = dataset["validation"]  # choose "train", "validation", or "test"
print(f"Loaded dataset split '{split}' with {len(split)} samples.")

# -------------------------
# Run Benchmark
# -------------------------
results = {
    "sentence": [],
    "ground_truth": [],
    "prediction": [],
    "correct": []
}

for item in split:
    sentence = item["sentence"]
    label = item["label"]  # 0 = negative, 1 = positive
    pred = classify(sentence)
    results["sentence"].append(sentence)
    results["ground_truth"].append(label)
    results["prediction"].append(pred)
    results["correct"].append(pred == label)


df = pd.DataFrame(results)
print("\nOverall Accuracy:", df["correct"].mean())


# three rounds: Overall Accuracy: 0.7534403669724771