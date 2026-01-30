import argparse
import json
import random
from pathlib import Path
 
import pandas as pd
 
try:
    from datasets import load_dataset
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'datasets'. Install with: pip install -r requirements.txt"
    ) from exc
 
 
def normalize_text(value):
    if value is None:
        return ""
    text = str(value).replace("\n", " ").strip()
    return " ".join(text.split())
 
 
def format_pair(prompt, response):
    prompt_text = normalize_text(prompt)
    response_text = normalize_text(response)
    if not prompt_text or not response_text:
        return None
    return {"text": f"User: {prompt_text}\nAssistant: {response_text}"}
 
 
def load_faq_pairs(csv_path, seed, eval_ratio):
    df = pd.read_csv(csv_path)
    if "Questions" not in df.columns or "Answers" not in df.columns:
        raise ValueError("Expected columns 'Questions' and 'Answers' in FAQ CSV.")
 
    pairs = []
    for _, row in df[["Questions", "Answers"]].dropna().iterrows():
        record = format_pair(row["Questions"], row["Answers"])
        if record:
            pairs.append(record)
 
    rng = random.Random(seed)
    rng.shuffle(pairs)
 
    split_idx = max(1, int(len(pairs) * (1 - eval_ratio)))
    return pairs[:split_idx], pairs[split_idx:]
 
 
def load_empathetic_dialogues():
    dataset = load_dataset("empathetic_dialogues")
    train_rows = dataset["train"]
    eval_rows = dataset["validation"]
 
    def rows_to_pairs(rows):
        pairs = []
        for row in rows:
            prompt = row.get("prompt") or row.get("context")
            response = row.get("utterance")
            if isinstance(prompt, list):
                prompt = " ".join(prompt)
            record = format_pair(prompt, response)
            if record:
                pairs.append(record)
        return pairs
 
    return rows_to_pairs(train_rows), rows_to_pairs(eval_rows)
 
 
def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            json.dump(record, handle, ensure_ascii=False)
            handle.write("\n")
 
 
def main():
    parser = argparse.ArgumentParser(
        description="Build a combined training dataset for GPT-2 fine-tuning."
    )
    parser.add_argument(
        "--faq",
        default="Data/Mental_Health_FAQ.csv",
        help="Path to the FAQ CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory to write combined JSONL outputs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for FAQ splitting.",
    )
    parser.add_argument(
        "--faq-eval-ratio",
        type=float,
        default=0.1,
        help="Fraction of FAQ data reserved for evaluation.",
    )
 
    args = parser.parse_args()
 
    faq_train, faq_eval = load_faq_pairs(args.faq, args.seed, args.faq_eval_ratio)
    empath_train, empath_eval = load_empathetic_dialogues()
 
    combined_train = faq_train + empath_train
    combined_eval = faq_eval + empath_eval
 
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
 
    write_jsonl(output_dir / "combined_train.jsonl", combined_train)
    write_jsonl(output_dir / "combined_eval.jsonl", combined_eval)
 
    print(f"Wrote {len(combined_train)} training samples.")
    print(f"Wrote {len(combined_eval)} evaluation samples.")
 
 
if __name__ == "__main__":
    main()
