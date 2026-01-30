import argparse
import json
import math
from pathlib import Path
 
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
 
 
def load_texts(path, max_samples):
    texts = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            text = record.get("text")
            if text:
                texts.append(text)
            if max_samples and len(texts) >= max_samples:
                break
    return texts
 
 
def compute_perplexity(texts, tokenizer, model, device):
    losses = []
    for text in texts:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        input_ids = encoded["input_ids"].to(device)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
        losses.append(outputs.loss.item())
 
    if not losses:
        return float("nan")
    return math.exp(sum(losses) / len(losses))
 
 
def main():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned GPT-2 model.")
    parser.add_argument("--model", required=True, help="Path to model directory.")
    parser.add_argument("--data", required=True, help="Path to JSONL evaluation file.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Maximum number of samples to evaluate.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device selection for evaluation.",
    )
    parser.add_argument(
        "--sample-prompt",
        default=None,
        help="Optional prompt to generate a sample response.",
    )
    args = parser.parse_args()
 
    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model path does not exist: {model_path}")
 
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"
    if args.device in {"cpu", "cuda"}:
        device = args.device
    device = torch.device(device)
 
    tokenizer = GPT2Tokenizer.from_pretrained(str(model_path))
    model = GPT2LMHeadModel.from_pretrained(str(model_path)).to(device)
    model.eval()
 
    texts = load_texts(args.data, args.max_samples)
    perplexity = compute_perplexity(texts, tokenizer, model, device)
    print(f"Perplexity: {perplexity:.2f}")
 
    if args.sample_prompt:
        encoded = tokenizer.encode(args.sample_prompt, return_tensors="pt").to(device)
        output = model.generate(
            encoded,
            max_new_tokens=128,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
        )
        print("\nSample generation:")
        print(tokenizer.decode(output[0], skip_special_tokens=True))
 
 
if __name__ == "__main__":
    main()
