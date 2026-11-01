"""
Train the SMS Spam Classifier and create fresh model.pkl and vectorizer.pkl.

Keep this file in the same folder as spam.csv.
Run: python train_model.py
"""

import pickle
import string
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "spam.csv"


def load_stop_words():
    """Use the NLTK stopwords already installed on this computer."""
    try:
        return set(stopwords.words("english"))
    except LookupError:
        print("NLTK English stopwords are missing.")
        print("Install them with: python -m nltk.downloader stopwords")
        raise


STOP_WORDS = load_stop_words()
STEMMER = PorterStemmer()


def transform_text(message):
    message = str(message).lower()
    # preserve_line=True avoids requiring the punkt_tab sentence model.
    tokens = nltk.word_tokenize(message, preserve_line=True)

    cleaned_tokens = []
    for token in tokens:
        if token.isalnum() and token not in STOP_WORDS and token not in string.punctuation:
            cleaned_tokens.append(STEMMER.stem(token))

    return " ".join(cleaned_tokens)


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_FILE.name}. Put spam.csv in: {BASE_DIR}"
        )

    df = pd.read_csv(DATA_FILE)

    # Dataset used in the notebook has columns v1 (label) and v2 (message).
    if not {"v1", "v2"}.issubset(df.columns):
        raise ValueError(
            "Expected spam.csv to contain columns named 'v1' and 'v2'. "
            f"Found: {list(df.columns)}"
        )

    df = df[["v1", "v2"]].rename(columns={"v1": "target", "v2": "text"})
    df = df.dropna(subset=["target", "text"])
    df = df.drop_duplicates(keep="first")

    df["target"] = df["target"].map({"ham": 0, "spam": 1})
    df = df.dropna(subset=["target"])
    df["target"] = df["target"].astype(int)

    print(f"Training messages: {len(df)}")
    print("Preprocessing messages...")
    transformed_messages = df["text"].apply(transform_text)

    # Evaluate on a held-out split first.
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        transformed_messages,
        df["target"],
        test_size=0.2,
        random_state=2,
        stratify=df["target"],
    )

    evaluation_vectorizer = TfidfVectorizer(max_features=3000)
    X_train = evaluation_vectorizer.fit_transform(X_train_text)
    X_test = evaluation_vectorizer.transform(X_test_text)

    evaluation_model = MultinomialNB()
    evaluation_model.fit(X_train, y_train)
    predictions = evaluation_model.predict(X_test)

    print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}")

    # Refit fresh artifacts on all available messages for the Streamlit app.
    final_vectorizer = TfidfVectorizer(max_features=3000)
    X_all = final_vectorizer.fit_transform(transformed_messages)

    final_model = MultinomialNB()
    final_model.fit(X_all, df["target"])

    with open(BASE_DIR / "vectorizer.pkl", "wb") as file:
        pickle.dump(final_vectorizer, file)

    with open(BASE_DIR / "model.pkl", "wb") as file:
        pickle.dump(final_model, file)

    print("\nDone! Fresh files created:")
    print(" - vectorizer.pkl")
    print(" - model.pkl")


if __name__ == "__main__":
    main()
