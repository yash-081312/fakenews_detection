"""
predict.py
Inference module for Fake News Detection.
Loads trained artifacts and classifies unseen news headlines/articles.
"""

import os
import sys

# Ensure root project directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
from src.preprocessor import clean_text

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(ROOT_DIR, "models")
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
DEFAULT_VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")


class FakeNewsClassifier:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, vectorizer_path: str = DEFAULT_VECTORIZER_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Run src/train.py first.")
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Vectorizer not found at {vectorizer_path}. Run src/train.py first.")

        print(f"[Inference] Loading model from {model_path}...")
        self.model = joblib.load(model_path)
        print(f"[Inference] Loading vectorizer from {vectorizer_path}...")
        self.vectorizer = joblib.load(vectorizer_path)
        self.feature_names = np.array(self.vectorizer.get_feature_names_out())

    def predict(self, raw_text: str) -> dict:
        """
        Classifies an input text as REAL or FAKE news.

        Returns a dictionary with:
        - raw_text: input text
        - cleaned_text: preprocessed text
        - prediction: 'FAKE' or 'REAL'
        - confidence: confidence percentage (float)
        - top_keywords: top TF-IDF words extracted from input
        """
        cleaned = clean_text(raw_text)
        if not cleaned:
            return {
                "raw_text": raw_text,
                "cleaned_text": "",
                "prediction": "UNKNOWN",
                "confidence": 0.0,
                "top_keywords": []
            }

        tfidf_vec = self.vectorizer.transform([cleaned])

        # Get class prediction
        pred_label_idx = self.model.predict(tfidf_vec)[0]
        label_str = "FAKE" if pred_label_idx == 1 else "REAL"

        # Estimate confidence
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(tfidf_vec)[0]
            confidence = float(probs[pred_label_idx]) * 100
        elif hasattr(self.model, "decision_function"):
            decision = self.model.decision_function(tfidf_vec)[0]
            # Logistic sigmoid transform
            prob_fake = 1 / (1 + np.exp(-decision))
            confidence = float(prob_fake if pred_label_idx == 1 else (1 - prob_fake)) * 100
        else:
            confidence = 99.0

        # Extract top keywords present in the text based on TF-IDF weight
        non_zero_indices = tfidf_vec.nonzero()[1]
        feature_scores = [(self.feature_names[i], tfidf_vec[0, i]) for i in non_zero_indices]
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        top_keywords = [word for word, score in feature_scores[:7]]

        return {
            "raw_text": raw_text,
            "cleaned_text": cleaned,
            "prediction": label_str,
            "confidence": round(confidence, 2),
            "top_keywords": top_keywords
        }


def run_interactive_cli():
    print("\n" + "="*60)
    print("   FAKE NEWS DETECTION - INTERACTIVE PREDICTION SYSTEM")
    print("="*60)
    print("Type or paste news text below. Enter 'exit' or 'quit' to stop.\n")

    classifier = FakeNewsClassifier()

    while True:
        try:
            user_input = input("\nEnter News Headline / Content:\n> ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting. Thank you!")
                break

            result = classifier.predict(user_input)
            badge = "[ FAKE NEWS ]" if result["prediction"] == "FAKE" else "[ REAL NEWS ]"
            color_hint = "RED ALERT" if result["prediction"] == "FAKE" else "VERIFIED"

            print("\n" + "-"*40)
            print(f"Result:     {badge} ({color_hint})")
            print(f"Confidence: {result['confidence']}%")
            print(f"Top terms:  {', '.join(result['top_keywords'])}")
            print("-"*40)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break


if __name__ == "__main__":
    classifier = FakeNewsClassifier()
    test_samples = [
        "WASHINGTON (Reuters) - U.S. Senate passes bipartisan budget deal after late-night negotiations.",
        "BREAKING BOMBSHELL: Proof that celebrities and politicians are using secret laser satellites to alter weather patterns!"
    ]
    for sample in test_samples:
        res = classifier.predict(sample)
        print(f"\nText: {res['raw_text']}")
        print(f"Prediction: {res['prediction']} ({res['confidence']}%)")
        print(f"Key Tokens: {res['top_keywords']}")
