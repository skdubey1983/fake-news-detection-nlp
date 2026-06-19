import re
import joblib
import nltk
import streamlit as st

from nltk.corpus import stopwords

nltk.download("stopwords")

MODEL_PATH = "models/fake_news_model.pkl"

stop_words = set(stopwords.words("english"))

CATEGORY_KEYWORDS = {
    "Politics": [
        "government", "minister", "election", "president", "prime minister",
        "parliament", "party", "vote", "policy", "law"
    ],
    "Health": [
        "doctor", "health", "cancer", "diabetes", "medicine", "hospital",
        "vaccine", "virus", "disease", "cure"
    ],
    "Technology": [
        "technology", "ai", "artificial intelligence", "software", "computer",
        "cyber", "data", "robot", "app", "internet"
    ],
    "Finance": [
        "bank", "stock", "market", "money", "rupee", "dollar",
        "economy", "tax", "loan", "investment"
    ],
    "Sports": [
        "cricket", "football", "match", "player", "team",
        "tournament", "score", "world cup", "olympic"
    ]
}


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def predict_news(news_text):
    model = load_model()
    cleaned_text = clean_text(news_text)

    prediction = model.predict([cleaned_text])[0]
    probability = model.predict_proba([cleaned_text])[0]

    # WELFake label mapping:
    # 0 = Real
    # 1 = Fake
    real_prob = probability[0]
    fake_prob = probability[1]

    return prediction, fake_prob, real_prob, probability, cleaned_text


def get_confidence_level(confidence):
    if confidence >= 0.85:
        return "High Confidence"
    elif confidence >= 0.70:
        return "Medium Confidence"
    else:
        return "Low Confidence"


def detect_category(text):
    text = text.lower()

    category_scores = {}

    for category, keywords in CATEGORY_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            if keyword in text:
                score += 1
        category_scores[category] = score

    best_category = max(category_scores, key=category_scores.get)

    if category_scores[best_category] == 0:
        return "General News"

    return best_category


def get_top_suspicious_words(cleaned_text, top_n=8):
    model = load_model()

    tfidf = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]

    feature_names = tfidf.get_feature_names_out()
    coefficients = classifier.coef_[0]

    word_score_map = dict(zip(feature_names, coefficients))

    words = cleaned_text.split()
    suspicious_words = []

    for word in words:
        if word in word_score_map:
            score = word_score_map[word]
            if score > 0:
                suspicious_words.append((word, score))

    suspicious_words = sorted(
        suspicious_words,
        key=lambda x: x[1],
        reverse=True
    )

    unique_words = []
    seen = set()

    for word, score in suspicious_words:
        if word not in seen:
            unique_words.append((word, score))
            seen.add(word)

    return unique_words[:top_n]


if "total_predictions" not in st.session_state:
    st.session_state.total_predictions = 0

if "fake_count" not in st.session_state:
    st.session_state.fake_count = 0

if "real_count" not in st.session_state:
    st.session_state.real_count = 0

if "confidence_sum" not in st.session_state:
    st.session_state.confidence_sum = 0.0


st.set_page_config(
    page_title="Fake News Detection using NLP",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Fake News Detection using NLP")
st.write("This app predicts whether a news headline/article is Fake or Real.")

st.divider()

st.subheader("📊 Prediction Dashboard")

col1, col2, col3, col4 = st.columns(4)

avg_confidence = 0
if st.session_state.total_predictions > 0:
    avg_confidence = (
        st.session_state.confidence_sum /
        st.session_state.total_predictions
    )

col1.metric("Total", st.session_state.total_predictions)
col2.metric("Fake", st.session_state.fake_count)
col3.metric("Real", st.session_state.real_count)
col4.metric("Avg Confidence", f"{avg_confidence * 100:.2f}%")

st.divider()

news_input = st.text_area(
    "Enter News Text:",
    height=220,
    placeholder="Paste news headline or article here..."
)

if st.button("Check News"):
    if news_input.strip() == "":
        st.warning("Please enter some news text.")
    else:
        prediction, fake_prob, real_prob, probability, cleaned_text = predict_news(news_input)

        confidence = max(fake_prob, real_prob)
        confidence_level = get_confidence_level(confidence)
        category = detect_category(news_input)

        st.session_state.total_predictions += 1
        st.session_state.confidence_sum += confidence

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("Prediction: FAKE NEWS")
            st.session_state.fake_count += 1
        else:
            st.success("Prediction: REAL NEWS")
            st.session_state.real_count += 1

        st.write(f"Fake Probability: {fake_prob * 100:.2f}%")
        st.write(f"Real Probability: {real_prob * 100:.2f}%")

        st.progress(float(confidence))

        st.info(f"Confidence Level: {confidence_level}")
        st.info(f"Detected Category: {category}")

        suspicious_words = get_top_suspicious_words(cleaned_text)

        if prediction == 1 and suspicious_words:
            st.subheader("Top Suspicious Words")

            for word, score in suspicious_words:
                st.write(f"• {word}")

        with st.expander("Debug Information"):
            st.write("Raw Prediction:", int(prediction))
            st.write("Raw Probabilities:", probability)
            st.write("Label Mapping: 0 = Real, 1 = Fake")
            st.write("Cleaned Text:", cleaned_text)