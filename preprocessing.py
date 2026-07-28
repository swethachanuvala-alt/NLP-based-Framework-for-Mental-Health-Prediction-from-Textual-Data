"""
preprocessing.py
-----------------
Text cleaning pipeline used to train the GloVe + GRU model.
This EXACT sequence of steps must be used at inference time too,
otherwise the tokenizer / embedding indices won't line up with
what the model was trained on.

Pipeline (mirrors the notebook):
  1. lowercase
  2. strip URLs
  3. strip HTML tags
  4. remove punctuation (keep apostrophes, so "don't" survives)
  5. collapse whitespace
  6. word_tokenize
  7. remove stopwords (but keep negation words - they matter for sentiment)
  8. lemmatize each token as a verb
"""

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def ensure_nltk_data():
    """Download the NLTK resources needed for cleaning, if not already present."""
    resources = {
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
    }
    for path, name in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


ensure_nltk_data()

_lemmatizer = WordNetLemmatizer()

_stop_words = set(stopwords.words("english"))
_KEEP_WORDS = {
    "not", "no", "nor", "never",
    "don't", "didn't", "doesn't",
    "can't", "won't", "isn't",
    "aren't", "wasn't", "weren't",
    "couldn't", "shouldn't", "wouldn't",
    "haven't", "hasn't", "hadn't",
}
_CUSTOM_STOPWORDS = _stop_words - _KEEP_WORDS


def clean_text(text: str) -> str:
    """Steps 1-5: lowercase, strip urls/html/punctuation, collapse whitespace."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^\w\s']", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_and_normalize(text: str) -> list:
    """Full pipeline: clean -> tokenize -> remove stopwords -> lemmatize.

    Returns a list of tokens, exactly what the Keras Tokenizer was
    fit on during training (it accepts pre-tokenized lists of words).
    """
    cleaned = clean_text(text)
    tokens = word_tokenize(cleaned)
    tokens = [t for t in tokens if t.lower() not in _CUSTOM_STOPWORDS]
    tokens = [_lemmatizer.lemmatize(t, pos="v") for t in tokens]
    return tokens


def preprocess_batch(texts) -> list:
    """Apply tokenize_and_normalize to an iterable of raw text strings."""
    return [tokenize_and_normalize(t) for t in texts]
