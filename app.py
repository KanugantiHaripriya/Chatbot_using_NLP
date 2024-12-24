import streamlit as st
import numpy as np
import tensorflow as tf
import random
import json
import nltk
import pickle
from nltk.stem import WordNetLemmatizer


# Load model and necessary files
lemmatizer = WordNetLemmatizer()

# Load the trained model
model = tf.keras.models.load_model('chatbot_model.h5')

# Load tokenized words and class labels
with open('words.pkl', 'rb') as file:
    words = pickle.load(file)

with open('classes.pkl', 'rb') as file:
    classes = pickle.load(file)

# Load intents file
with open('intents.json') as file:
    intents = json.load(file)

# Function to clean and tokenize user input
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Function to create bag-of-words representation
def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag)

# Function to predict the class of user input
def predict_class(sentence, model):
    bag = bow(sentence, words)
    res = model.predict(np.array([bag]))[0]
    ERROR_THRESHOLD = 0.25  # Confidence threshold
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

# Function to generate a response based on the predicted class
def get_response(intents_list, intents_json):
    if intents_list:
        tag = intents_list[0]['intent']
        for i in intents_json['intents']:
            if i['tag'] == tag:
                return random.choice(i['responses'])
    return "I'm sorry, I don't understand that."

# Streamlit UI
st.title("Chat Connect")
st.write("Where Conversations Meet Intelligence!")

# User input section
user_input = st.text_input("Ask Away!:", "")

if user_input:
    # Predict the intent and generate a response
    predicted_intents = predict_class(user_input, model)
    response = get_response(predicted_intents, intents)  # Pass the JSON file as the second argument
    st.write(f"Bot: {response}")




