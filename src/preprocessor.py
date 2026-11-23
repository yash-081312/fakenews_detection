"""
preprocessor.py
Text preprocessing pipeline for Fake News Detection.
Includes normalization, cleaning, tokenization, stopword removal, and lemmatization.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Standard 179 NLTK English stopwords fallback
DEFAULT_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "ain", "all", "am", "an", "and", "any", "are", "aren",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can", "couldn", "couldn't", "d", "did", "didn", "didn't", "do", "does", "doesn", "doesn't", "doing", "don",
    "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn", "hadn't", "has", "hasn",
    "hasn't", "have", "haven", "haven't", "having", "he", "her", "here", "hers", "herself", "him", "himself",
    "his", "how", "i", "if", "in", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "ll",
    "m", "ma", "me", "mightn", "mightn't", "more", "most", "mustn", "mustn't", "my", "myself", "needn", "needn't",
    "no", "nor", "not", "now", "o", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves",
    "out", "over", "own", "re", "s", "same", "shan", "shan't", "she", "she's", "should", "should've", "shouldn",
    "shouldn't", "so", "some", "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until",
    "up", "ve", "very", "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "when", "where",
    "which", "while", "who", "whom", "why", "will", "with", "won", "won't", "wouldn", "wouldn't", "y", "you",
    "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
}

# Ensure required NLTK resources are available with safe instant fallback
try:
    STOPWORDS = set(stopwords.words('english'))
except Exception:
    STOPWORDS = DEFAULT_STOPWORDS

class DummyLemmatizer:
    def lemmatize(self, word):
        return word

try:
    LEMMATIZER = WordNetLemmatizer()
    LEMMATIZER.lemmatize("testing")
except Exception:
    LEMMATIZER = DummyLemmatizer()

# Precompile regular expressions for optimal speed
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
HTML_PATTERN = re.compile(r'<.*?>')
NON_ASCII_PATTERN = re.compile(r'[^\x00-\x7F]+')
PUNCT_DIGIT_PATTERN = re.compile(r'[^a-zA-Z\s]')
EXTRA_WHITESPACE_PATTERN = re.compile(r'\s+')


def clean_text(text: str, remove_stopwords: bool = True, lemmatize: bool = True) -> str:
    """
    Cleans raw input text through a multi-stage NLP pipeline.

    Args:
        text (str): Raw text string (headline or article body).
        remove_stopwords (bool): Whether to filter out standard English stopwords.
        lemmatize (bool): Whether to perform word lemmatization.

    Returns:
        str: Cleaned and normalized text string.
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = URL_PATTERN.sub(' ', text)

    # 3. Remove HTML markup tags
    text = HTML_PATTERN.sub(' ', text)

    # 4. Remove non-ASCII characters
    text = NON_ASCII_PATTERN.sub(' ', text)

    # 5. Remove digits and punctuation (retain alphabet letters and whitespace)
    text = PUNCT_DIGIT_PATTERN.sub(' ', text)

    # 6. Tokenize by whitespace
    tokens = text.split()

    # 7. Remove stopwords & short words (< 2 characters)
    if remove_stopwords:
        tokens = [token for token in tokens if token not in STOPWORDS and len(token) > 2]

    # 8. Lemmatization
    if lemmatize:
        tokens = [LEMMATIZER.lemmatize(token) for token in tokens]

    # 9. Rejoin tokens into clean string
    return " ".join(tokens)


def batch_clean_texts(texts, show_progress: bool = False):
    """
    Cleans a collection of texts efficiently.
    """
    if show_progress:
        try:
            from tqdm import tqdm
            return [clean_text(t) for t in tqdm(texts, desc="Cleaning texts")]
        except ImportError:
            pass
    return [clean_text(t) for t in texts]


if __name__ == "__main__":
    sample_text = (
        "BREAKING NEWS: Donald Trump just tweeted https://t.co/xyz123 that elections are RIGGED! "
        "Read <b>more</b> at www.fakenews.com."
    )
    print("Raw text:")
    print(sample_text)
    print("\nCleaned text:")
    print(clean_text(sample_text))
