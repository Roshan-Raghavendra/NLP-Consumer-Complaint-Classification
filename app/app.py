import streamlit as st
import joblib
import re
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk

# Download required NLTK resources
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "complaint_classifier.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"

# --------------------------------------------------
# Load model and vectorizer
# --------------------------------------------------

model = joblib.load(MODEL_PATH)
tfidf = joblib.load(VECTORIZER_PATH)

# --------------------------------------------------
# NLP preprocessing
# --------------------------------------------------

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):

    text = str(text).lower()

    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# --------------------------------------------------
# Sentiment analysis
# --------------------------------------------------

sentiment_analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):

    scores = sentiment_analyzer.polarity_scores(text)

    if scores["compound"] >= 0.05:
        sentiment = "Positive"
    elif scores["compound"] <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return sentiment, scores["compound"]


# --------------------------------------------------
# Priority prediction
# --------------------------------------------------

def predict_priority(text, sentiment):

    text_lower = text.lower()

    critical_keywords = [
        "fraud",
        "stolen",
        "unauthorized",
        "identity theft",
        "scam",
        "emergency"
    ]

    high_keywords = [
        "urgent",
        "immediately",
        "wrong charge",
        "incorrect",
        "blocked",
        "cannot access"
    ]

    if any(keyword in text_lower for keyword in critical_keywords):
        return "Critical"

    elif sentiment == "Negative" and any(
        keyword in text_lower for keyword in high_keywords
    ):
        return "High"

    elif sentiment == "Negative":
        return "Medium"

    else:
        return "Low"


# --------------------------------------------------
# Complete analysis
# --------------------------------------------------

def analyze_complaint(text):

    processed_text = preprocess_text(text)

    vector = tfidf.transform([processed_text])

    category = model.predict(vector)[0]

    sentiment, sentiment_score = analyze_sentiment(text)

    priority = predict_priority(text, sentiment)

    return category, sentiment, sentiment_score, priority


# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------

st.set_page_config(
    page_title="Consumer Complaint Analyzer",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Consumer Complaint Analyzer")

st.write(
    "Enter a consumer complaint to automatically classify "
    "the complaint category, sentiment and priority."
)

complaint = st.text_area(
    "Enter Complaint",
    height=180,
    placeholder="Example: I have an unauthorized transaction on my account..."
)

if st.button("Analyze Complaint"):

    if complaint.strip() == "":
        st.warning("Please enter a complaint.")

    else:

        category, sentiment, sentiment_score, priority = analyze_complaint(
            complaint
        )

        st.subheader("Analysis Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Complaint Category", category)
            st.metric("Sentiment", sentiment)

        with col2:
            st.metric("Priority", priority)
            st.metric("Sentiment Score", round(sentiment_score, 3))

        st.success("Complaint analyzed successfully.")