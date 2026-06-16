import re
import torch
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from model import SpamClassifier


def clean_text(text):
    text = str(text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    return text.split()

def vectorize_text(text, vocab):
    vector = torch.zeros(len(vocab))

    words = clean_text(text)

    for word in words:
        if word in vocab:
            vector[vocab[word]] = 1
    return vector


@st.cache_resource
def load_model():
    checkpoint = torch.load("spam_model.pt", map_location="cpu")
    vocab = checkpoint["vocab"]
    model = SpamClassifier(vocab_size=len(vocab))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, vocab

def predict(text, model, vocab):
    x = vectorize_text(text, vocab).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        spam_probability = torch.sigmoid(logits).item()

    return spam_probability

st.title("Youtube Comment Spam Detector")
st.write("Paste a comment and the model will predict if it is spam or not.")

try:
    model, vocab = load_model()
except Exception as e:
    st.error("Model failed to load.")
    st.exception(e)
    st.stop()


comment = st.text_input("Enter a comment")
if st.button("Analyze Comment"):
    if comment.strip():
        spam_prob = predict(comment, model, vocab)
        normal_prob = 1 - spam_prob

        if spam_prob >= 0.5:
            st.error(f"Prediction: Spam ({spam_prob:.1%})")
        else:
            st.success(f"Prediction: Not spam ({normal_prob:.1%})")

        st.write("Spam probability:", round(spam_prob, 3))
    else:
        st.warning("Type a comment first.")


st.divider()

st.header("Analyze a CSV of comments")

uploaded_file = st.file_uploader("Upload a CSV 'text'", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "text" not in df.columns:
        st.error("CSV needs to be named 'text'")
    else:
        df["spam_probability"] = df["text"].apply(lambda x: predict(str(x), model, vocab))
        df["prediction"] = df["spam_probability"].apply(
            lambda x: "Spam" if x >= 0.5 else "Normal"
        )

        st.dataframe(df)

        st.subheader("Prediction Counts")
        counts = df["prediction"].value_counts()

        fig, ax = plt.subplots()
        ax.bar(counts.index, counts.values)
        ax.set_ylabel("Number of comments")
        st.pyplot(fig)
