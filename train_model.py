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
RESULTS_DIR = "results"
DATA_DIR = "data"

MODEL_PATH = os.path.join(MODEL_DIR, "fake_news_model.pkl")
RESULTS_PATH = os.path.join(RESULTS_DIR, "dataset_comparison_results.csv")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

stop_words = set(stopwords.words("english"))


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)


def create_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2)
        )),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ])


def load_welfake_dataset():
    print("\nLoading WELFake Dataset...")

    dataset = load_dataset("davanstrien/WELFake")
    df = dataset["train"].to_pandas()

    print("\nWELFake Columns:")
    print(df.columns)

    print("\nWELFake Shape:")
    print(df.shape)

    print("\nWELFake Label Counts:")
    print(df["label"].value_counts())

    df = df.dropna()

    df["content"] = (
        df["title"].astype(str)
        + " "
        + df["text"].astype(str)
    )

    df["content"] = df["content"].apply(clean_text)

    # WELFake:
    # 0 = Real
    # 1 = Fake
    X = df["content"]
    y = df["label"].astype(int)

    return X, y


def load_liar_local_dataset():
    print("\nChecking Local LIAR Dataset...")

    liar_dir = os.path.join(DATA_DIR, "LIAR_dataset")

    train_path = os.path.join(liar_dir, "train.tsv")
    valid_path = os.path.join(liar_dir, "valid.tsv")
    test_path = os.path.join(liar_dir, "test.tsv")

    if not (
        os.path.exists(train_path)
        and os.path.exists(valid_path)
        and os.path.exists(test_path)
    ):
        print("\nLIAR dataset not found.")
        print("Expected files:")
        print("data/LIAR_dataset/train.tsv")
        print("data/LIAR_dataset/valid.tsv")
        print("data/LIAR_dataset/test.tsv")
        return None, None

    print("\nLoading Local LIAR Dataset...")

    columns = [
        "id",
        "label",
        "statement",
        "subject",
        "speaker",
        "speaker_job",
        "state",
        "party",
        "barely_true_counts",
        "false_counts",
        "half_true_counts",
        "mostly_true_counts",
        "pants_on_fire_counts",
        "context"
    ]

    train_df = pd.read_csv(train_path, sep="\t", header=None, names=columns)
    valid_df = pd.read_csv(valid_path, sep="\t", header=None, names=columns)
    test_df = pd.read_csv(test_path, sep="\t", header=None, names=columns)

    df = pd.concat([train_df, valid_df, test_df], ignore_index=True)

    print("\nLIAR Columns:")
    print(df.columns)

    print("\nLIAR Shape:")
    print(df.shape)

    print("\nLIAR Original Label Counts:")
    print(df["label"].value_counts())

    # LIAR original labels:
    # true, mostly-true, half-true, barely-true, false, pants-fire
    #
    # Binary mapping used here:
    # Real = true, mostly-true, half-true
    # Fake = barely-true, false, pants-fire

    real_labels = ["true", "mostly-true", "half-true"]
    fake_labels = ["barely-true", "false", "pants-fire"]

    def map_liar_label(label):
        label = str(label).strip().lower()

        if label in real_labels:
            return 0
        if label in fake_labels:
            return 1

        return None

    df["binary_label"] = df["label"].apply(map_liar_label)

    df = df.dropna(subset=["statement", "binary_label"])

    print("\nLIAR Binary Label Counts:")
    print(df["binary_label"].value_counts())

    df["content"] = df["statement"].astype(str).apply(clean_text)

    X = df["content"]
    y = df["binary_label"].astype(int)

    return X, y


def load_kaggle_fake_real_dataset():
    print("\nChecking Kaggle Fake/Real News Dataset...")

    fake_path = os.path.join(DATA_DIR, "Fake.csv")
    true_path = os.path.join(DATA_DIR, "True.csv")

    if not os.path.exists(fake_path) or not os.path.exists(true_path):
        print("\nKaggle dataset not found.")
        print("To use it, download Fake.csv and True.csv and place them in data/ folder.")
        return None, None

    print("\nLoading Kaggle Fake/Real Dataset...")

    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    fake_df["label"] = 1
    true_df["label"] = 0

    df = pd.concat([fake_df, true_df], ignore_index=True)
    df = df.dropna()

    print("\nKaggle Columns:")
    print(df.columns)

    print("\nKaggle Shape:")
    print(df.shape)

    print("\nKaggle Label Counts:")
    print(df["label"].value_counts())

    if "title" in df.columns and "text" in df.columns:
        df["content"] = (
            df["title"].astype(str)
            + " "
            + df["text"].astype(str)
        )
    elif "text" in df.columns:
        df["content"] = df["text"].astype(str)
    else:
        raise ValueError("Kaggle dataset must contain a text column.")

    df["content"] = df["content"].apply(clean_text)

    X = df["content"]
    y = df["label"].astype(int)

    return X, y


def train_and_evaluate(dataset_name, X, y, save_model=False):
    print("\n" + "=" * 60)
    print(f"Training on Dataset: {dataset_name}")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = create_model()

    print("\nTraining started...")
    model.fit(X_train, y_train)
    print("Training completed.")

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\nModel Evaluation")
    print("----------------")
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Real", "Fake"],
            zero_division=0
        )
    )

    print("\nModel Classes:")
    print(model.named_steps["classifier"].classes_)

    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Real", "Fake"]
    )

    disp.plot()
    plt.title(f"Confusion Matrix - {dataset_name}")
    plt.tight_layout()

    cm_file_name = f"confusion_matrix_{dataset_name.lower().replace(' ', '_')}.png"
    cm_path = os.path.join(RESULTS_DIR, cm_file_name)

    plt.savefig(cm_path)
    plt.close()

    print(f"\nConfusion matrix saved at: {cm_path}")

    if save_model:
        joblib.dump(model, MODEL_PATH)
        print(f"\nMain model saved at: {MODEL_PATH}")

    return {
        "Dataset": dataset_name,
        "Model": "TF-IDF + Logistic Regression",
        "Accuracy (%)": round(accuracy * 100, 2),
        "Precision (%)": round(precision * 100, 2),
        "Recall (%)": round(recall * 100, 2),
        "F1 Score (%)": round(f1 * 100, 2),
        "Total Samples": len(X),
        "Train Samples": len(X_train),
        "Test Samples": len(X_test)
    }


def main():
    all_results = []

    X_welfake, y_welfake = load_welfake_dataset()

    result_welfake = train_and_evaluate(
        dataset_name="WELFake",
        X=X_welfake,
        y=y_welfake,
        save_model=True
    )

    all_results.append(result_welfake)

    X_liar, y_liar = load_liar_local_dataset()

    if X_liar is not None and y_liar is not None:
        result_liar = train_and_evaluate(
            dataset_name="LIAR",
            X=X_liar,
            y=y_liar,
            save_model=False
        )
        all_results.append(result_liar)

    X_kaggle, y_kaggle = load_kaggle_fake_real_dataset()

    if X_kaggle is not None and y_kaggle is not None:
        result_kaggle = train_and_evaluate(
            dataset_name="Kaggle Fake Real",
            X=X_kaggle,
            y=y_kaggle,
            save_model=False
        )
        all_results.append(result_kaggle)

    results_df = pd.DataFrame(all_results)

    print("\n" + "=" * 60)
    print("Final Dataset Comparison")
    print("=" * 60)
    print(results_df)

    results_df.to_csv(RESULTS_PATH, index=False)

    print(f"\nComparison results saved at: {RESULTS_PATH}")


if __name__ == "__main__":
    main()