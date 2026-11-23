"""
news_analyzer.py
Comprehensive analysis engine for Fake News Detection:
- Computes Fake % vs Real / Authenticity %
- Extracts "What News is Circulated" (core claim, circulating topics, entities)
- Identifies linguistic red flags and credible journalistic signals
"""

import os
import sys
import re
from collections import Counter

# Ensure root project directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import joblib
from src.preprocessor import clean_text

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "best_model.pkl")
VECTORIZER_PATH = os.path.join(ROOT_DIR, "models", "tfidf_vectorizer.pkl")

# Characteristic signal dictionaries
SENSATIONALIST_TRIGGERS = {
    "shocking", "bombshell", "breaking", "exposed", "conspiracy", "secret",
    "hoax", "busted", "unbelievable", "mindblowing", "leaked", "rigged",
    "stabs", "pathetic", "destroy", "traitor", "hidden", "alien", "cheating",
    "banned", "censored", "mainstream media", "deep state", "plot", "scam"
}

JOURNALISTIC_TRIGGERS = {
    "reuters", "associated press", "spokesman", "spokeswoman", "statement",
    "official", "confirmed", "department", "spokesperson", "reported",
    "according to", "representative", "investigation", "analysis", "published",
    "research", "findings", "legislation", "committee", "briefing", "wire"
}


class NewsCirculationAnalyzer:
    def __init__(self, model_path: str = MODEL_PATH, vectorizer_path: str = VECTORIZER_PATH):
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            raise FileNotFoundError("Model or Vectorizer file missing. Please ensure models are trained.")
        
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.feature_names = np.array(self.vectorizer.get_feature_names_out())

    def analyze_news(self, text: str, source_metadata: dict = None) -> dict:
        """
        Performs full deep-dive news detection & circulation analysis.
        """
        if not text or len(text.strip()) == 0:
            return {"error": "Input text is empty."}

        cleaned = clean_text(text)
        tfidf_vec = self.vectorizer.transform([cleaned])

        # 1. Probability Estimation
        # In our model: 1 = Fake, 0 = Real
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(tfidf_vec)[0]
            prob_real = float(probs[0])
            prob_fake = float(probs[1])
        elif hasattr(self.model, "decision_function"):
            decision = float(self.model.decision_function(tfidf_vec)[0])
            prob_fake = 1 / (1 + np.exp(-decision))
            prob_real = 1 - prob_fake
        else:
            pred = self.model.predict(tfidf_vec)[0]
            prob_fake = 0.99 if pred == 1 else 0.01
            prob_real = 1 - prob_fake

        fake_percentage = round(prob_fake * 100, 2)
        real_percentage = round(prob_real * 100, 2)

        # Confidence level
        confidence = round(max(fake_percentage, real_percentage), 2)

        # 2. Risk Classification Verdict
        if fake_percentage >= 70:
            verdict = "HIGH RISK OF FAKE NEWS"
            verdict_badge = "🔴 FAKE / FABRICATED"
            status_color = "#e74c3c"
            risk_level = "High"
        elif fake_percentage >= 40:
            verdict = "SUSPICIOUS / UNVERIFIED CONTENT"
            verdict_badge = "🟡 MIXED / SUSPICIOUS"
            status_color = "#f39c12"
            risk_level = "Moderate"
        else:
            verdict = "CREDIBLE & AUTHENTIC NEWS"
            verdict_badge = "🟢 VERIFIED / REAL"
            status_color = "#2ecc71"
            risk_level = "Low"

        # 3. What News is Being Circulated? (Summary & Key Topics)
        circulation_analysis = self._extract_circulating_story(text, cleaned)

        # 4. Extract Top TF-IDF Weighted Features
        non_zero_indices = tfidf_vec.nonzero()[1]
        feature_scores = [(self.feature_names[i], float(tfidf_vec[0, i])) for i in non_zero_indices]
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        top_salient_terms = [word for word, score in feature_scores[:10]]

        # 5. Linguistic Signals (Red Flags vs Credibility Anchors)
        lower_raw = text.lower()
        red_flags_found = [term for term in SENSATIONALIST_TRIGGERS if re.search(r'\b' + re.escape(term) + r'\b', lower_raw)]
        credibility_anchors = [term for term in JOURNALISTIC_TRIGGERS if re.search(r'\b' + re.escape(term) + r'\b', lower_raw)]

        # Sensationalism Metrics
        exclamation_count = text.count("!")
        words = text.split()
        unique_words = len(set(w.lower() for w in words))
        lexical_diversity = round(unique_words / max(1, len(words)), 3)
        caps_words = [w for w in words if w.isupper() and len(w) > 2]
        caps_ratio = round((len(caps_words) / max(1, len(words))) * 100, 1)

        result = {
            "verdict": verdict,
            "verdict_badge": verdict_badge,
            "status_color": status_color,
            "risk_level": risk_level,
            "fake_percentage": fake_percentage,
            "real_percentage": real_percentage,
            "confidence": confidence,
            "accuracy_confidence": confidence,
            "circulation_summary": circulation_analysis["summary"],
            "circulating_topics": circulation_analysis["key_entities"],
            "top_salient_terms": top_salient_terms,
            "red_flags_found": red_flags_found,
            "credibility_anchors": credibility_anchors,
            "sensationalism_metrics": {
                "exclamation_count": exclamation_count,
                "all_caps_words": caps_words[:5],
                "all_caps_percentage": caps_ratio,
                "total_word_count": len(words),
                "lexical_diversity": lexical_diversity
            },
            "source_metadata": source_metadata or {"source_type": "Direct Text Input"}
        }

        return result

    def _extract_circulating_story(self, raw_text: str, cleaned_text: str) -> dict:
        """
        Summarizes what news is being circulated and extracts main entities/topics.
        """
        # Extract first 2-3 sentences for concise summary
        sentences = re.split(r'(?<=[.!?])\s+', raw_text.strip())
        summary = " ".join(sentences[:3])
        if len(summary) > 350:
            summary = summary[:347] + "..."

        # Extract potential named entities / capitalized phrases from raw text
        candidate_entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', raw_text)
        # Filter common words
        common_words = {"The", "This", "That", "There", "Here", "It", "They", "We", "You", "He", "She", "In", "On", "At", "By", "For", "With", "About", "Against", "Between", "Into", "Through", "During", "Before", "After", "Above", "Below", "To", "From", "Up", "Down", "In", "Out", "Off", "Over", "Under", "Again", "Further", "Then", "Once"}
        filtered_entities = [e for e in candidate_entities if e not in common_words and len(e) > 3]

        entity_counts = Counter(filtered_entities).most_common(6)
        key_entities = [item[0] for item in entity_counts]

        if not key_entities:
            # Fallback to top cleaned words
            key_entities = cleaned_text.split()[:5]

        return {
            "summary": summary,
            "key_entities": key_entities
        }


if __name__ == "__main__":
    analyzer = NewsCirculationAnalyzer()
    test_text = (
        "WASHINGTON (Reuters) - The White House on Wednesday confirmed that congressional leaders "
        "have drafted a bipartisan legislation to fund government operations through September."
    )
    res = analyzer.analyze_news(test_text)
    print("Verdict:", res["verdict"])
    print(f"Fake %: {res['fake_percentage']}% | Real %: {res['real_percentage']}%")
    print("Circulating Topics:", res["circulating_topics"])
