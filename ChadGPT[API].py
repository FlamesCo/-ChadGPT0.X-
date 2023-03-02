import streamlit as st
import transformers
import torch

# Set up the Streamlit app
st.set_page_config(page_title="Chatbot", page_icon=":speech_balloon:")
st.title("Chatbot")

# Define a function to generate a response using the specified model
def generate_response(user_input, model):
    inputs = tokenizer(user_input, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=1024)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Initialize the conversation history with a greeting
conversation_history = "Hello! I'm a chatbot. How can I help you today?"

# Define a function to handle the "Send" button click
def send_button_click():
    global conversation_history
    user_input = input_box.value.strip()
    if user_input:
        conversation_history += "\n\nUser: " + user_input
        response = generate_response(user_input, model)
        conversation_history += "\n\nChatbot: " + response
        chat_history_area.text(conversation_history)
        input_box.value = ""

# Check if TensorFlow 2.0 or PyTorch is installed
if "tensorflow" in transformers.__file__:
    # Use TensorFlow 2.0 and the GPT-2 model
    import tensorflow as tf
    from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = TFGPT2LMHeadModel.from_pretrained("gpt2")
elif "torch" in transformers.__file__:
    # Use PyTorch and the DistilGPT2 model
    from transformers import DistilGPT2Tokenizer, DistilGPT2Model
    tokenizer = DistilGPT2Tokenizer.from_pretrained("distilgpt2")
    model = DistilGPT2Model.from_pretrained("distilgpt2")
    model.eval()
else:
    # Raise an error if neither TensorFlow 2.0 nor PyTorch is installed
    raise RuntimeError("At least one of TensorFlow 2.0 or PyTorch should be installed.")

# Create a text input box for the user to enter messages
input_box = st.text_input("Type your message here...", key="input_box")

# Create a button to send messages
send_button = st.button("Send", on_click=send_button_click)

# Create an area to display the chat history
chat_history_area = st.empty()
chat_history_area.text(conversation_history)
