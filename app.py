import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Load the trained model
MODEL_PATH = "C:\\Users\\kavin\\Desktop\\Intent\\intent_classification (2).h5"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Load Tokenizer (You may need to save & load it from training)
MAX_SEQUENCE_LENGTH = 100  # Same value used in training
MAX_WORDS = 10000

tokenizer = Tokenizer(num_words=MAX_WORDS)  # Ensure this matches training
label_encoder = LabelEncoder()

# Sample intent labels (Replace with actual labels from training)
LABELS = [
    "GetWeather", "PlayMusic", "SearchCreativeWork", "BookRestaurant",
    "AddToPlaylist", "RateBook", "SearchScreeningEvent", "Excitement",
    "Cancellation", "Greetings", "Affirmation", "BookMeeting"
]

label_encoder.fit(LABELS)  # Simulating fitted label encoder

# Function to preprocess user input
def preprocess_text(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="post")
    return np.array(padded_sequences)

# Function to predict intent
def predict_intent(text):
    processed_text = preprocess_text(text)
    prediction = model.predict(processed_text)
    predicted_label_index = np.argmax(prediction)
    predicted_intent = label_encoder.inverse_transform([predicted_label_index])[0]
    return predicted_intent

# Streamlit UI
st.title("Intent Classification App")
st.write("Enter a sentence, and the model will predict the intent.")

user_input = st.text_input("Enter your text:")

if st.button("Predict"):
    if user_input:
        prediction = predict_intent(user_input)
        st.success(f"Predicted Intent: {prediction}")
    else:
        st.warning("Please enter a sentence.")
