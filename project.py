# project.py — CS116 Sentiment (single-file VS Code version)

# ====== Imports ======
import os
import pickle
from pathlib import Path
import urllib.request
import re
import random

import pandas as pd
import numpy as np

# plots are optional; comment out if you don't need them
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    ConfusionMatrixDisplay,
)

# ====== Config ======
DATA_DIR = Path("data")
MODELS_DIR = Path("models")
PARQUET_PATH = DATA_DIR / "0000.parquet"
HF_URL = ("https://huggingface.co/datasets/stanfordnlp/sentiment140/"
          "resolve/refs%2Fconvert%2Fparquet/sentiment140/train/0000.parquet")

# start small for speed; set to None to use full dataset (heavy)
SAMPLE_N = 50_000

# reproducibility
RNG = 42

# ====== Setup ======
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# make sure NLTK stopwords are available
try:
    _ = stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

STOP = set(stopwords.words("english"))
STEM = SnowballStemmer("english")


# ====== Utilities ======
def download_data_if_missing():
    if PARQUET_PATH.exists():
        print("Found dataset:", PARQUET_PATH)
        return
    print("Downloading dataset from HuggingFace...")
    urllib.request.urlretrieve(HF_URL, PARQUET_PATH)
    print("Saved to", PARQUET_PATH.resolve())


def load_raw():
    # file has columns: sentiment, ids, date, query, text
    df = pd.read_parquet(PARQUET_PATH)
    # keep only binary labels 0/4
    df = df[df["sentiment"].isin([0, 4])].copy()
    if SAMPLE_N is not None and len(df) > SAMPLE_N:
        df = df.sample(SAMPLE_N, random_state=RNG)
    print("Loaded rows:", len(df))
    return df[["sentiment", "text"]]


def clean_tweet(text, stem=False):
    # lower
    text = (text or "").lower()
    # remove @mentions
    text = re.sub(r"@\w+", " ", text)
    # remove urls
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    # keep letters/digits/space only
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # remove stopwords; optional stemming
    cleaned_words = []
    for w in text.split():
        if w not in STOP:
            cleaned_words.append(STEM.stem(w) if stem else w)
    return " ".join(cleaned_words)


def explore_quick(df):
    # label distribution
    print("Total tweets:", len(df))
    counts = df["sentiment"].value_counts()
    print(counts)

    sns.set_style("whitegrid")
    sns.barplot(
        x=["Negative", "Positive"],
        y=[counts.get(0, 0), counts.get(4, 0)],
        hue=["Negative", "Positive"],
    )
    plt.title("Distribution of Sentiment Labels")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Tweets")
    plt.show(block=False)

    # optional quick wordclouds on raw text (comment out if slow)
    # neg_text = " ".join(df[df["sentiment"] == 0]["text"])
    # pos_text = " ".join(df[df["sentiment"] == 4]["text"])
    # for txt, bg, title in [(neg_text, "black", "Negative"), (pos_text, "white", "Positive")]:
    #     wc = WordCloud(width=800, height=400, background_color=bg).generate(txt)
    #     plt.figure(figsize=(10, 5)); plt.imshow(wc, interpolation="bilinear")
    #     plt.axis("off"); plt.title(title); plt.show(block=False)


def vectorize_and_split(df_clean):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df_clean["text"])
    y = df_clean["sentiment"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RNG
    )
    return X_train, X_test, y_train, y_test, vectorizer


def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000, solver="liblinear", verbose=1)
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy on test set:", acc)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # --- delete or comment these 4 lines so it doesn't pop a window ---
    # ConfusionMatrixDisplay.from_predictions(
    #     y_test, y_pred, display_labels=["Negative", "Positive"], cmap="Blues"
    # )
    # plt.grid(False)
    # plt.title("Confusion Matrix")
    # plt.show(block=False)
    # ---------------------------------------------------------------

    return y_pred



def show_top_words(model, vectorizer, N=10):
    feats = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]
    order = np.argsort(coefs)
    top_neg = feats[order[:N]]
    top_pos = feats[order[-N:]]
    print("Top positive words:\n", list(top_pos))
    print("Top negative words:\n", list(top_neg))


def save_artifacts(model, vectorizer):
    with open(MODELS_DIR / "sentiment_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(MODELS_DIR / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    print("Saved models to", MODELS_DIR.resolve())


def load_artifacts():
    with open(MODELS_DIR / "sentiment_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODELS_DIR / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


def predict_sentiment(tweet, model, vectorizer):
    cleaned = clean_tweet(tweet)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    return "Negative ☹️" if pred == 0 else "Positive 😊"


def tweeter(model, vectorizer):
    s = input('Enter a tweet to analyze (or "log off"): ')
    while s != "log off":
        print("Your tweet was", predict_sentiment(s, model, vectorizer))
        s = input('Enter a tweet to analyze (or "log off"): ')


# ====== Main flow (run this file) ======
if __name__ == "__main__":
    # 1) get data
    download_data_if_missing()
    df = load_raw()

    # 2) quick EDA (optional)
    # explore_quick(df)

    # 3) clean
    print("Cleaning text...")
    df["text"] = df["text"].astype(str).apply(clean_tweet)
    df = df[df["text"].str.strip().astype(bool)]
    print("After cleaning rows:", len(df))

    # 4) vectorize + split
    print("Vectorizing and splitting...")
    X_train, X_test, y_train, y_test, vectorizer = vectorize_and_split(df)

    # 5) train
    print("Training Logistic Regression...")
    model = train_model(X_train, y_train)

    # 6) evaluate
    _ = evaluate(model, X_test, y_test)
    show_top_words(model, vectorizer, N=10)

    # 7) save
    save_artifacts(model, vectorizer)

    # 8) load & predict examples
    loaded_model, loaded_vectorizer = load_artifacts()
    examples = [
        "I absolutely love the new design of your app!",
        "I absolutely HATE!!! DESIGN!!! RECIPE!!!!.",
        "Not sure how I feel about our exam... it was interesting, I guess.",
        "I hate design recipe, but the python content was great!",
    ]
    for ex in examples:
        print(ex, "->", predict_sentiment(ex, loaded_model, loaded_vectorizer))

    # 9) interactive loop (uncomment to use)
    # tweeter(loaded_model, loaded_vectorizer)

# ====== Visualizer (ADD-ONLY) ===============================================
# Drop this at the very end of project.py (after your current code)
from pathlib import Path as _Path
import matplotlib.pyplot as _plt
import seaborn as _sns
import numpy as _np
from sklearn.metrics import ConfusionMatrixDisplay as _CMD

def _save_fig(fig, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


def make_and_save_visuals(df_clean, model, vectorizer, X_test, y_test):
    base = _Path(__file__).parent
    plots = base / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    # 1) Confusion Matrix
    y_pred_for_plot = model.predict(X_test)
    fig_cm, ax_cm = _plt.subplots(figsize=(6, 5))
    _CMD.from_predictions(
        y_test, y_pred_for_plot,
        display_labels=["Negative", "Positive"],
        cmap="Blues", ax=ax_cm
    )
    ax_cm.grid(False)
    ax_cm.set_title("Confusion Matrix")
    _save_fig(fig_cm, plots / "confusion_matrix.png")

    # 2) Label Distribution (after cleaning)
    counts = df_clean["sentiment"].value_counts()
    fig_dist, ax_dist = _plt.subplots(figsize=(6, 4))
    _sns.set_style("whitegrid")
    _sns.barplot(
        x=["Negative", "Positive"],
        y=[counts.get(0, 0), counts.get(4, 0)],
        hue=["Negative", "Positive"],
        ax=ax_dist
    )
    ax_dist.set_xlabel("Sentiment")
    ax_dist.set_ylabel("Number of Tweets")
    ax_dist.set_title("Distribution of Sentiment Labels (cleaned)")
    ax_dist.legend().remove()
    _save_fig(fig_dist, plots / "label_distribution.png")

    # 3) Top words bar charts (same words you print)
    feats = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]
    order = _np.argsort(coefs)
    top_neg = feats[order[:10]]
    top_neg_w = coefs[order[:10]]
    top_pos = feats[order[-10:]]
    top_pos_w = coefs[order[-10:]]

    # Positive
    fig_tp, ax_tp = _plt.subplots(figsize=(7, 4))
    ax_tp.barh(list(top_pos), list(top_pos_w))
    ax_tp.set_title("Top Positive Words (weights)")
    ax_tp.set_xlabel("Weight")
    _save_fig(fig_tp, plots / "top_words_positive.png")

    # Negative
    fig_tn, ax_tn = _plt.subplots(figsize=(7, 4))
    ax_tn.barh(list(top_neg), list(top_neg_w))
    ax_tn.set_title("Top Negative Words (weights)")
    ax_tn.set_xlabel("Weight")
    _save_fig(fig_tn, plots / "top_words_negative.png")

    # 4) (optional) word clouds on cleaned text
    try:
        from wordcloud import WordCloud as _WordCloud
        pos_text = " ".join(df_clean[df_clean["sentiment"] == 4]["text"])
        neg_text = " ".join(df_clean[df_clean["sentiment"] == 0]["text"])

        for txt, name, bg in [(pos_text, "wordcloud_positive.png", "white"),
                              (neg_text, "wordcloud_negative.png", "black")]:
            wc = _WordCloud(width=1000, height=500, background_color=bg).generate(txt)
            _plt.figure(figsize=(10, 5))
            _plt.imshow(wc, interpolation="bilinear")
            _plt.axis("off")
            _plt.title(name.replace(".png","").replace("_"," ").title())
            _plt.savefig(plots / name, dpi=150, bbox_inches="tight")
            print(f"Saved {plots / name}")
    except Exception as e:
        print("Wordcloud step skipped:", e)

# === RUN VISUALIZER (create+save all plots, then pop windows) ===
make_and_save_visuals(df, model, vectorizer, X_test, y_test)
plt.show()






