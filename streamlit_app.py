import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

@st.cache_data
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("tokenizer/content/tokenizer/september_28")
    model = AutoModelForTokenClassification.from_pretrained("trainer/content/trainer/september_28")
    return tokenizer, model

tokenizer, model = load_model()

# Create a text input field for the user to enter text
text = st.text_input('Enter some text')

# Check if the text input is not empty
if text:
    # Create a NER pipeline
    nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="average")

    # Make predictions
    predictions = nlp(text)

    # Display the predictions
    st.write(predictions)
