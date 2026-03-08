"""
nlp/summarizer.py
Generates an AI-written structured summary report using OpenAI API.
Falls back to a rule-based summary if no API key is provided.
"""

from collections import defaultdict


def generate_summary(text: str, analysis: dict, lang: str = "en", api_key: str = None) -> str:
    """
    Generate a structured crime news summary report.
    Uses OpenAI GPT-4o if API key provided, otherwise uses rule-based fallback.
    """
    if api_key:
        return _openai_summary(text, analysis, lang, api_key)
    else:
        return _rule_based_summary(text, analysis, lang)


def _openai_summary(text: str, analysis: dict, lang: str, api_key: str) -> str:
    """Generate summary using OpenAI API."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        # Build context from NLP analysis
        entities = analysis.get("entities", [])
        grouped = defaultdict(list)
        for ent in entities:
            grouped[ent["label"]].append(ent["text"])

        people = list(set(grouped.get("PERSON", [])))[:5]
        locations = list(set(grouped.get("GPE", []) + grouped.get("LOC", [])))[:5]
        orgs = list(set(grouped.get("ORG", [])))[:5]
        crime_types = list(analysis.get("crime_types", {}).keys())
        sentiment = analysis.get("sentiment", {})

        lang_instruction = {
            "he": "The article is in Hebrew. Respond in English.",
            "ar": "The article is in Arabic. Respond in English.",
        }.get(lang, "")

        prompt = f"""You are a crime news analyst. Analyze this news article and produce a structured intelligence report.

{lang_instruction}

ARTICLE TEXT:
{text[:3000]}

NLP PRE-ANALYSIS:
- Sentiment: {sentiment.get('label', 'N/A')} ({sentiment.get('score', 0):.1%} confidence)
- People mentioned: {', '.join(people) if people else 'None detected'}
- Locations: {', '.join(locations) if locations else 'None detected'}
- Organizations: {', '.join(orgs) if orgs else 'None detected'}
- Crime categories: {', '.join(crime_types) if crime_types else 'None detected'}

Write a structured report with these exact sections:
## 📋 Executive Summary
(2-3 sentence overview of the incident)

## 🔑 Key Facts
(Bullet points: who, what, where, when, how)

## 📍 Locations Involved
(List all locations mentioned and their relevance)

## 👥 Key Individuals & Organizations
(Brief note on each named person/organization)

## ⚠️ Crime Classification
(Type of crime, severity assessment)

## 🔍 Analyst Notes
(Context, patterns, significance)

Be factual and objective. Do not speculate beyond what the article states."""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.3
        )
        return response.choices[0].message.content

    except Exception as e:
        return _rule_based_summary(text, analysis, lang) + f"\n\n*(OpenAI error: {str(e)})*"


def _rule_based_summary(text: str, analysis: dict, lang: str) -> str:
    """
    Generate a structured summary without an API key.
    Uses the NLP analysis results to build a report.
    """
    entities = analysis.get("entities", [])
    grouped = defaultdict(list)
    for ent in entities:
        grouped[ent["label"]].append(ent["text"])

    people = list(set(grouped.get("PERSON", [])))[:5]
    locations = list(set(grouped.get("GPE", []) + grouped.get("LOC", [])))[:5]
    orgs = list(set(grouped.get("ORG", [])))[:5]
    dates = list(set(grouped.get("DATE", [])))[:3]
    crime_types = analysis.get("crime_types", {})
    sentiment = analysis.get("sentiment", {})
    keywords = analysis.get("keywords", [])[:10]
    word_count = analysis.get("word_count", 0)

    lang_label = {"en": "English", "he": "Hebrew", "ar": "Arabic"}.get(lang, "Unknown")

    lines = [
        "## 📋 Executive Summary",
        f"This {lang_label}-language article ({word_count} words) covers a news story with "
        f"**{sentiment.get('label', 'neutral')}** sentiment "
        f"({sentiment.get('score', 0):.1%} confidence). "
        f"The article mentions {len(entities)} named entities across {len(locations)} locations.",
        "",
        "## 🔑 Key Facts",
        f"- **Word count:** {word_count}",
        f"- **Language:** {lang_label}",
        f"- **Sentiment:** {sentiment.get('label', 'N/A').capitalize()} ({sentiment.get('score', 0):.1%})",
        f"- **Top keywords:** {', '.join(keywords[:7]) if keywords else 'N/A'}",
        "",
        "## 📍 Locations Involved",
    ]

    if locations:
        for loc in locations:
            lines.append(f"- {loc}")
    else:
        lines.append("- No locations detected")

    lines += ["", "## 👥 Key Individuals & Organizations"]
    if people:
        lines.append("**People:**")
        for p in people:
            lines.append(f"- {p}")
    if orgs:
        lines.append("**Organizations:**")
        for o in orgs:
            lines.append(f"- {o}")
    if not people and not orgs:
        lines.append("- No named individuals or organizations detected")

    lines += ["", "## ⚠️ Crime Classification"]
    if crime_types:
        for category, keywords_found in crime_types.items():
            lines.append(f"- **{category.capitalize()}** (keywords: {', '.join(keywords_found[:3])})")
    else:
        lines.append("- No specific crime categories detected")

    if dates:
        lines += ["", "## 📅 Dates Mentioned"]
        for d in dates:
            lines.append(f"- {d}")

    lines += [
        "",
        "## 🔍 Analyst Notes",
        f"*Add your OpenAI API key in the sidebar to generate an AI-powered analysis of this article.*",
        f"*Rule-based summary generated from {len(entities)} detected entities.*"
    ]

    return "\n".join(lines)
