# Chatbot using NLP

This project implements a chatbot using Natural Language Processing (NLP) to understand user input, classify intents, and provide appropriate responses.

---

## Features
- **Intent Recognition:** Classifies user queries into predefined intents.
- **Custom Responses:** Provides tailored responses based on the detected intent.
- **Model Training:** Trains a neural network to understand text data.
- **Interactive Interface:** Supports integration with frameworks like Streamlit.

---

## Setup and Usage

### Clone the Repository
```bash
git clone https://github.com/yourusername/chatbot-using-nlp.git
```


## File Structure

- **intents.json**: Defines the intents, patterns, and responses.
- **train_chatbot.py**: Script to preprocess data and train the model.
- **app.py**: Streamlit application for interacting with the chatbot.
- **chatbot_model.h5**: Trained chatbot model.
- **words.pkl** & **classes.pkl**: Saved data for words and classes.


---

## Requirements
The following Python libraries are required to run the project:

- **TensorFlow**: Neural network training and inference.
- **NLTK**: Natural language preprocessing.
- **Streamlit**: For building the chatbot interface.
- **NumPy**: Numerical computations.
- **Scikit-Learn**: For text vectorization.

# Interaction Example

Run the chatbot application using the following command:

```bash
streamlit run app.py


