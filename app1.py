import streamlit as st
from transformers import pipeline

# Project Title and Objectives
st.title("SentimentSense: AI-Powered Emotion Detection for E-Commerce Reviews")

st.header("Project Objectives")
objectives = [
    "Develop an AI-Powered Emotion Detection Model - Train a DistilBERT-based model to classify emotions (Anger, Doubt, Frustration, Satisfaction, Trust) from Amazon reviews.",
    "Enhance Sentiment Analysis with Deep Learning - Utilize transformers and NLP techniques to extract meaningful insights beyond traditional sentiment analysis (positive/negative).",
    "Handle Imbalanced Data Effectively - Implement class weighting techniques to ensure fair predictions across all emotion categories.",
    "Optimize Model Performance for Real-World Use - Fine-tune tokenization, batch processing, and model hyperparameters to improve accuracy and efficiency.",
    "Develop an Interactive Chatbot for Emotion Prediction - Create a user-friendly chatbot in Google Colab to allow dynamic emotion prediction from customer reviews.",
    "Enable Business Applications in Customer Experience Management - Provide actionable insights for e-commerce businesses to enhance product strategies and customer engagement."
]
for obj in objectives:
    st.markdown(f"- {obj}")

# Load the emotion detection model
@st.cache_resource  # Cache the model to avoid reloading every time
def load_model():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

emotion_detector = load_model()

# Define emotion mapping based on the model's labels
emotion_mapping = {
    "anger": "Anger",
    "joy": "Joy",
    "fear": "Fear",
    "sadness": "Sadness",
    "surprise": "Surprise",
    "neutral": "Neutral"
}

# Streamlit UI
st.header("Amazon Review Emotion Detector Chatbot")
st.write("Analyze customer emotions from Amazon reviews.")

# Dropdown for Category Selection
categories = ['books', 'mobile', 'smartTv', 'refrigerator', 'mobile accessories']
category = st.selectbox("Select Category:", categories)

# Dropdown for Rating Selection
ratings = ['1', '2', '3', '4', '5']
rating = st.selectbox("Select Rating:", ratings)

# Text Input for Review
review = st.text_area("Enter Your Review:", placeholder="Type your review here...")

# Submit Button
if st.button("Submit"):
    if review.strip():
        # Construct query and get prediction
        query_text = f"Category: {category}, Rating: {rating}, Review: {review}"
        result = emotion_detector(query_text)

        # Get the predicted label and confidence score
        predicted_label = result[0]['label']
        predicted_score = result[0]['score']

        # Map the label to emotion
        predicted_emotion = emotion_mapping.get(predicted_label.lower(), "Unknown Emotion")

        # Display results
        st.subheader("Predicted Emotion:")
        st.write(f"**{predicted_emotion}**")
        st.write(f"**Model Confidence:** {predicted_score:.4f}")
    else:
        st.warning("Please enter a review before submitting.")

# Refresh Button
if st.button("Refresh"):
    st.experimental_rerun()  # Refresh the app
