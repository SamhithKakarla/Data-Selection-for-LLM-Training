import torch
from transformers import AutoTokenizer
from model import TinyGPT

# --- Config ---
CKPT = './tiny_gpt_runs/model_epoch3.pt'
TOKENIZER = 'gpt2'
MAX_LEN = 64
D_MODEL = 128
N_LAYERS = 4
N_HEADS = 4
NUM_CLASSES = 2
LABELS = {0: 'Negative', 1: 'Positive'}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

# --- Model ---
model = TinyGPT(
    vocab_size=tokenizer.vocab_size,
    max_len=MAX_LEN,
    d_model=D_MODEL,
    n_layers=N_LAYERS,
    n_heads=N_HEADS,
    num_classes=NUM_CLASSES
).to(device)

model.load_state_dict(torch.load(CKPT, map_location=device))
model.eval()

# --- Prediction Utils ---
def _encode(texts):
    enc = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=MAX_LEN,
        return_tensors='pt'
    )
    return enc['input_ids'].to(device), enc['attention_mask'].to(device)

def predict_sentiment(text):
    ids, mask = _encode([text])
    with torch.no_grad():
        logits = model(ids, attention_mask=mask)
        probs = torch.softmax(logits, dim=-1)[0]
    pred = probs.argmax().item()
    return pred, probs[pred].item(), probs.cpu().numpy()

def predict_batch(texts):
    ids, mask = _encode(texts)
    with torch.no_grad():
        logits = model(ids, attention_mask=mask)
        probs = torch.softmax(logits, dim=-1)
    preds = probs.argmax(dim=-1)
    confs = probs.gather(1, preds.unsqueeze(1)).squeeze(1)
    return preds.cpu().numpy(), confs.cpu().numpy(), probs.cpu().numpy()

# --- Example ---
text = "This movie is absolutely fantastic!"
pred, conf, probs = predict_sentiment(text)
print(f'"{text}" → {LABELS[pred]} ({conf:.4f})')
for i, p in enumerate(probs):
    print(f'  {LABELS[i]}: {p:.4f}')

texts = [
    "This movie is great!",
    "This could be worse",
    "It was okay, nothing special.",
    "Absolutely loved it!",
    "Worst product ever."
]

preds, confs, _ = predict_batch(texts)
for t, p, c in zip(texts, preds, confs):
    print(f'"{t}" → {LABELS[p]} ({c:.4f})')