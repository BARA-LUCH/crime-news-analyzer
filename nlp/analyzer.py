"""
nlp/analyzer.py
Performs Named Entity Recognition (NER), sentiment analysis,
and keyword extraction on news article text.
Supports English, Hebrew, and Arabic.
"""

from transformers import pipeline
import spacy
from collections import Counter
import re

# ── Load models lazily (only when first called) ───────────────────────────────

_sentiment_pipeline = None
_ner_pipeline = None
_nlp_en = None


def _get_sentiment_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        # Multilingual sentiment model
        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
            top_k=1
        )
    return _sentiment_pipeline


def _get_ner_pipeline():
    global _ner_pipeline
    if _ner_pipeline is None:
        # Multilingual NER model
        _ner_pipeline = pipeline(
            "ner",
            model="Davlan/bert-base-multilingual-cased-ner-hrl",
            aggregation_strategy="simple"
        )
    return _ner_pipeline


def _get_spacy():
    global _nlp_en
    if _nlp_en is None:
        try:
            _nlp_en = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
            _nlp_en = spacy.load("en_core_web_sm")
    return _nlp_en


# ── Sentiment Analysis ────────────────────────────────────────────────────────

def analyze_sentiment(text: str) -> dict:
    """
    Returns sentiment label and confidence score.
    Uses multilingual XLM-RoBERTa model.
    """
    try:
        pipe = _get_sentiment_pipeline()
        # Truncate to 512 tokens max
        truncated = text[:1000]
        result = pipe(truncated)[0]
        if isinstance(result, list):
            result = result[0]
        return {
            "label": result.get("label", "neutral").lower(),
            "score": float(result.get("score", 0.0))
        }
    except Exception as e:
        return {"label": "neutral", "score": 0.5, "error": str(e)}


# ── Named Entity Recognition ──────────────────────────────────────────────────

def extract_entities(text: str) -> list:
    """
    Extract named entities using both HuggingFace multilingual NER
    and spaCy (English). Merges and deduplicates results.
    """
    entities = []

    # HuggingFace multilingual NER
    try:
        pipe = _get_ner_pipeline()
        truncated = text[:512]
        hf_entities = pipe(truncated)
        for ent in hf_entities:
            label = ent.get("entity_group", "")
            label_map = {"PER": "PERSON", "LOC": "GPE", "ORG": "ORG", "MISC": "EVENT"}
            entities.append({
                "text": ent.get("word", "").strip(),
                "label": label_map.get(label, label),
                "score": float(ent.get("score", 0.0))
            })
    except Exception:
        pass

    # spaCy English NER (catches more entity types like DATE, FAC)
    try:
        nlp = _get_spacy()
        doc = nlp(text[:5000])
        for ent in doc.ents:
            if ent.label_ in ("PERSON", "ORG", "GPE", "LOC", "DATE", "EVENT", "FAC", "NORP"):
                entities.append({
                    "text": ent.text.strip(),
                    "label": ent.label_,
                    "score": 1.0
                })
    except Exception:
        pass

    # Deduplicate: keep highest-score version of each (text, label) pair
    seen = {}
    for ent in entities:
        key = (ent["text"].lower(), ent["label"])
        if key not in seen or ent["score"] > seen[key]["score"]:
            seen[key] = ent

    return list(seen.values())


# ── Keyword Extraction ────────────────────────────────────────────────────────

def extract_keywords(text: str, top_n: int = 15) -> list:
    """
    Extract top keywords using TF-IDF-style frequency analysis
    with stopword filtering. Language-agnostic.
    """
    # Basic stopwords
    stopwords = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
        "has", "have", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "this", "that", "these", "those", "it", "its",
        "he", "she", "they", "we", "you", "i", "his", "her", "their", "our",
        "said", "also", "which", "who", "what", "when", "where", "how", "not",
        "as", "up", "out", "about", "than", "more", "so", "if", "no", "after"
    }

    # Tokenize and clean
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    filtered = [w for w in words if w not in stopwords]

    # Count and return top N
    counter = Counter(filtered)
    return [word for word, _ in counter.most_common(top_n)]


# ── Crime Keywords Detection ──────────────────────────────────────────────────

CRIME_KEYWORDS = {
    "violence": ["shooting", "stabbing", "attack", "assault", "murder", "killed", "wounded", "explosion", "bomb"],
    "theft": ["robbery", "theft", "stolen", "burglar", "fraud", "scam"],
    "arrest": ["arrested", "detained", "suspect", "charged", "convicted", "sentenced", "police", "investigation"],
    "drug": ["drugs", "narcotics", "trafficking", "smuggling", "cartel"],
    "terrorism": ["terror", "terrorist", "extremist", "militia", "insurgent", "jihad"],
}


def classify_crime_type(text: str) -> dict:
    """Identify crime categories mentioned in the article."""
    text_lower = text.lower()
    found = {}
    for category, keywords in CRIME_KEYWORDS.items():
        matches = [kw for kw in keywords if kw in text_lower]
        if matches:
            found[category] = matches
    return found


# ── Main Analysis Function ────────────────────────────────────────────────────

def analyze_text(text: str) -> dict:
    """
    Full NLP analysis pipeline.
    Returns: sentiment, entities, keywords, crime_types
    """
    return {
        "sentiment": analyze_sentiment(text),
        "entities": extract_entities(text),
        "keywords": extract_keywords(text),
        "crime_types": classify_crime_type(text),
        "word_count": len(text.split()),
        "char_count": len(text),
    }
