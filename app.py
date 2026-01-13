import streamlit as st
import numpy as np
import joblib
import re

model = joblib.load("mlmodel.pkl")
tfidf = joblib.load("tfidf.pkl")
le = joblib.load("le.pkl")

st.set_page_config()

st.title("Cuisines Classifier")

st.caption("this app helps you to predic a restaurants")

def clean_text(text):
    text =text.lower()
    text = re.sub('[^a-zA-Z, ]', '', text)
    return text

user_input = st.text_area('Cuisine(s)', placeholder="Example: North indian, chines")

if st.button('Predict Restaurant'):
    if user_input.strip() == "": 
        st.warning("Please enter cuisine !")
    else:
        cleaned = clean_text(user_input)
        vector = tfidf.transform([cleaned])
        prediction = model.predict(vector)
        restaurant = le.inverse_transform(prediction)
        st.success(f"Predicted Restaurant: **{restaurant[0]}**")