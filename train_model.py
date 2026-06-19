import os
import re
import joblib
import nltk
import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_dataset
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

nltk.download("stopwords")

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "fake_news_model.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

stop_words = set(stopwords.words("english"))


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)


def load_online_dataset():
    dataset = load_dataset("davanstrien/WELFake")

    df = dataset["train"].to_pandas()

    print("Dataset columns:", df.columns)
    print("Dataset shape:", df.shape)

    df = df.dropna()

    df["content"] = df["title"].astype(str) + " " + df["text"].astype(str)

    df["content"] = df["content"].apply(clean_text)

    return df["content"], df["label"]


def train():
    X, y = load_online_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2)
        )),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1 Score :", f1_score(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()

    joblib.dump(model, MODEL_PATH)
    print("Model saved at:", MODEL_PATH)


if __name__ == "__main__":
    train()