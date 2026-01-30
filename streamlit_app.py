import os
from pathlib import Path
 
import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
 
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "./my_model"))
 
 
@st.cache_resource
def load_model(model_path: Path):
    tokenizer = GPT2Tokenizer.from_pretrained(str(model_path))
    model = GPT2LMHeadModel.from_pretrained(str(model_path))
    return tokenizer, model
 
 
def generate_text(prompt, tokenizer, model):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs,
        max_new_tokens=128,
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
        num_return_sequences=1,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
 
 
st.title("Your Mental Health Supporter")
st.warning(
    "Not medical advice. This is a proof-of-concept demo and cannot replace "
    "professional help. If you are in crisis, please contact local emergency "
    "services or a trusted hotline."
)
 
if not MODEL_PATH.exists():
    st.error(
        "Model not found. Download the fine-tuned model and place it in "
        f"'{MODEL_PATH}/' or set MODEL_PATH."
    )
    st.stop()
 
tokenizer, model = load_model(MODEL_PATH)
 
user_input = st.text_area("How can I help you?", "")
 
if st.button("Generate"):
    with st.spinner("Thinking..."):
        if not user_input.strip():
            st.warning("Please enter a question to continue.")
            st.stop()
        generated_text = generate_text(user_input, tokenizer, model)
        if "history" not in st.session_state:
            st.session_state["history"] = []
        st.session_state["history"].append((user_input, generated_text))
 
if "history" in st.session_state:
    for i, (prompt, response) in enumerate(st.session_state["history"], 1):
        st.text_area(f"Your Question {i}", prompt, height=75)
        st.text_area(f"My Response {i}", response, height=150)
 
col1, col2 = st.columns(2)
with col1:
    if col1.button("👍 Useful", key="like_button"):
        st.session_state["like_button_disabled"] = True
        st.session_state["like"] = st.session_state.get("like", 0) + 1
        st.write("Thanks for your feedback!")
with col2:
    if col2.button("👎 Not Useful"):
        st.session_state["dislike"] = st.session_state.get("dislike", 0) + 1
        st.write("Sorry to hear that, I will try to be better next time")
