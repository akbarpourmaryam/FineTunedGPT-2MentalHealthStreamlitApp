import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load fine-tuned model and tokenizer
model_name_or_path = "./my_model"
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
model = GPT2LMHeadModel.from_pretrained(model_name_or_path)

def generate_text(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, num_return_sequences=1)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

# Streamlit app layout
st.title("Your Mental Health Supporter")

user_input = st.text_area("How can I help you?", "")

if st.button("Generate"):
    with st.spinner('Thinking...'):
        generated_text = generate_text(user_input)
        if 'history' not in st.session_state:
            st.session_state['history'] = []
        st.session_state['history'].append((user_input, generated_text))

if 'history' in st.session_state:
    for i, (prompt, response) in enumerate(st.session_state['history'], 1):
        st.text_area(f"Your Question {i}", prompt, height=75)
        st.text_area(f"My Response {i}", response, height=150)

# Like and Dislike buttons
col1, col2 = st.columns(2)
with col1:
    if col1.button('👍 Useful', key='like_button'):
        # Disable or hide the button after clicking
        st.session_state['like_button_disabled'] = True
        st.session_state['like'] = st.session_state.get('like', 0) + 1
        st.write("Thanks for your feedback!")
with col2:
    if col2.button('👎 Not Useful'):
        st.session_state['dislike'] = st.session_state.get('dislike', 0) + 1
        st.write("Sorry to hear that, I will try to be better next time")
