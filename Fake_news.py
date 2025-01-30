import streamlit as st
import pickle  # For loading the saved model and vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and vectorizer
model_path = "model.pkl"  # Path to your trained model file
vectorizer_path = "pipe.pkl"  # Path to your vectorizer file

# Load the model and vectorizer
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model or vectorizer: {e}")
    st.stop()  # Stop the app if there is an error loading the model/vectorizer

# Streamlit App UI
st.title("Real or Fake News Detector")
st.write("Upload your news headline or content to verify its authenticity.")

# User Input
user_input = st.text_area("Enter news content:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.error("Please enter some text.")
    else:
        try:
            # Transform the input text
            transformed_input = vectorizer.transform([user_input])
            
            # Make the prediction
            prediction = model.predict(transformed_input)
            
            # Show the result
            result = "Real News" if prediction[0] == 1 else "Fake News"
            st.success(f"The news is classified as: {result}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
