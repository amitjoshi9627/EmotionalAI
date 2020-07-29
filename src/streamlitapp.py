from sentiment_classifier import get_prediction
import streamlit as st

text = st.text_input("Enter Your Text..")
result = get_prediction(text)
if result:
    prediction = "Positive Sentiment"
else:
    prediction = "Negative Sentiment"
if text != "":
    st.write(prediction)