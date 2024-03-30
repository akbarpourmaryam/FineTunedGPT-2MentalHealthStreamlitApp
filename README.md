## Fine Tuned GPT-2 for Mental Health - Streamlit App
This project involves fine-tuning a GPT-2 model on a mental health FAQ dataset to generate text related to mental health. The fine-tuning process adapts the model to better understand and respond to queries in the domain of mental health.

# Repository Structure

- `Data/`: This directory contains the dataset used to train and fine-tune the GPT-2 model.
- `MentalHealthFAQ_FineTuner.ipynb`: A Jupyter notebook that details the process of fine-tuning the GPT-2 model on the mental health FAQ dataset.
- `data_exploratory_analysis.ipynb`: A Jupyter notebook that provides an exploratory analysis of the mental health FAQ dataset, offering insights into the data used for training the chatbot model.
- `streamlit_app.py`: The Streamlit application script that can be used to run the chatbot interface, allowing users to ask questions and receive responses generated by the fine-tuned model.
  
# Set up and Run the App
1. **Clone the Repository**: First, clone this repository to your local machine using Git. Open a terminal or command prompt and run the following command:
   ```bash
   git clone https://github.com/akbarpourmaryam/FineTunedGPT-2MentalHealthStreamlitApp)

2. ** Download the fine-tuned GPT-2 model: From the Releases section of this repository, download the fine-tuned GPT-2 model. Unzip the file and place it in the root directory of the cloned repository or update the streamlit_app.py file to point to the location where you have saved the model files.

3. **Install Streamlit and Transformers
   ```bash
   pip install streamlit transformers
   
4. **Run the Streamlit App
   ```bash
   streamlit run app.py
   
After running the Streamlit app, your web browser should automatically open the app, or you can navigate to the local URL provided in the terminal.

## Notes
The app.py script expects the fine-tuned GPT-2 model to be located in a folder named my_model in the same directory as the script.

# Data Source

The data used in this project comes from the "Mental Health FAQ for Chatbot" dataset at the following URL:
https://www.kaggle.com/datasets/narendrageek/mental-health-faq-for-chatbot/data


   
