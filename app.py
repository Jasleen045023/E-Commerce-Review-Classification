import streamlit as st
from transformers import pipeline

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
st.title("Amazon Review Emotion Detector Chatbot")
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
