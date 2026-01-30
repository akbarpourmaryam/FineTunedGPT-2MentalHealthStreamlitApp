# Fine-Tuned GPT-2 Mental Health FAQ (Streamlit Demo)

This is a proof-of-concept project that demonstrates an end-to-end ML workflow
(EDA -> fine-tune -> deploy). It fine-tunes GPT-2 on a mental health FAQ dataset
plus a supportive dialog dataset, then serves a Streamlit app for interactive
queries.

## Important disclaimer

This project is for educational purposes only and is not medical advice. The
model can generate incorrect or unsafe responses. If you or someone you know is
in crisis, contact local emergency services or a trusted hotline.

## Repository structure

- Data/                           Raw mental health FAQ dataset
- scripts/                        Data prep and evaluation helpers
- MentalHealthFAQ_FineTuner.ipynb Fine-tuning notebook
- data_exploratory_analysis.ipynb EDA notebook
- streamlit_app.py                Streamlit demo app
- requirements.txt                Python dependencies

## Quickstart

1) Clone the repository:

git clone https://github.com/akbarpourmaryam/FineTunedGPT-2MentalHealthStreamlitApp.git
cd FineTunedGPT-2MentalHealthStreamlitApp

2) Create a virtual environment and install dependencies:

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

3) Download the fine-tuned model:

Download the model from:
https://github.com/akbarpourmaryam/FineTunedGPT-2MentalHealthStreamlitApp/releases/latest

Unzip it into ./my_model, or set MODEL_PATH (see below).

4) Run the Streamlit app:

streamlit run streamlit_app.py

## Model download and expected folder structure

Place the model files under ./my_model by default:

my_model/
  config.json
  pytorch_model.bin
  vocab.json
  merges.txt
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json

## Environment variables

- MODEL_PATH: Path to the fine-tuned model directory (default: ./my_model)

## Data sources

- Mental Health FAQ for Chatbot (Kaggle):
  https://www.kaggle.com/datasets/narendrageek/mental-health-faq-for-chatbot/data
- EmpatheticDialogues (Hugging Face Datasets):
  https://huggingface.co/datasets/empathetic_dialogues

## Data prep and fine-tuning

Use the helper script to build a combined dataset:

python scripts/build_training_data.py --faq Data/Mental_Health_FAQ.csv --output-dir data

Then open MentalHealthFAQ_FineTuner.ipynb as a starting point for training.

## Evaluation

A lightweight evaluation script computes perplexity on a held-out set:

python scripts/evaluate_model.py --model ./my_model --data data/combined_eval.jsonl --max-samples 200

This evaluation is a baseline for monitoring training quality. For production
systems, add human review and safety checks.

## Sample outputs (model output will vary)

Prompt:
What should I do if I feel overwhelmed every day?

Response:
It can help to reach out to someone you trust and describe how you are feeling.
Small, consistent steps like sleep, hydration, and a short walk can reduce
overwhelm. If this feeling is intense or persistent, consider speaking to a
health professional.

Prompt:
Is it normal to feel anxious before exams?

Response:
Yes, it is common to feel anxious before exams. Preparing in advance, taking
breaks, and using simple breathing exercises can make the anxiety more
manageable.

## Notes

- This is a proof-of-concept demo, not a clinical tool.
- Model outputs depend on fine-tuning data quality and training settings.
git status -sb

git status -sb
