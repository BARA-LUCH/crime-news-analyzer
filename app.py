import streamlit as st
import tempfile
import os
from ocr.extractor import extract_text_from_image, extract_text_from_pdf
from nlp.analyzer import analyze_text
from nlp.summarizer import generate_summary
from map.visualizer import build_map
from utils.language import detect_language

st.set_page_config(
    page_title="AI Crime News Analyzer",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 AI Crime News Analyzer")
st.markdown("Upload a **news article screenshot or PDF** — get instant NLP analysis, a crime hotspot map, and an AI summary. Supports **English, Hebrew & Arabic**.")

st.divider()

# ── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    openai_key = st.text_input("OpenAI API Key", type="password", help="Required for AI summary generation")
    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown("1. Upload image or PDF")
    st.markdown("2. OCR extracts the text")
    st.markdown("3. NLP finds entities & sentiment")
    st.markdown("4. Locations plotted on map")
    st.markdown("5. AI generates summary report")
    st.markdown("---")
    st.markdown("Built by **Bara Luch**")
    st.markdown("[GitHub](https://github.com/bara-luch) | [LinkedIn](https://linkedin.com/in/bara-luch)")

# ── Upload ────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📁 Upload Article")
    uploaded_file = st.file_uploader(
        "Choose an image or PDF",
        type=["png", "jpg", "jpeg", "pdf"],
        help="Screenshot or PDF of a news article"
    )

    if uploaded_file:
        file_ext = uploaded_file.name.split(".")[-1].lower()
        st.success(f"✅ Uploaded: `{uploaded_file.name}`")

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        if file_ext == "pdf":
            if st.button("🚀 Analyze Article", type="primary", use_container_width=True):
                st.session_state["run"] = True
                st.session_state["path"] = tmp_path
                st.session_state["type"] = "pdf"
        else:
            st.image(tmp_path, caption="Uploaded Image", use_column_width=True)
            if st.button("🚀 Analyze Article", type="primary", use_container_width=True):
                st.session_state["run"] = True
                st.session_state["path"] = tmp_path
                st.session_state["type"] = "image"

with col2:
    st.subheader("✍️ Or Paste Text")
    pasted_text = st.text_area(
        "Paste article text directly",
        height=200,
        placeholder="Paste any news article text here — English, Hebrew, or Arabic..."
    )
    if st.button("🚀 Analyze Pasted Text", type="primary", use_container_width=True):
        if pasted_text.strip():
            st.session_state["run"] = True
            st.session_state["path"] = None
            st.session_state["type"] = "text"
            st.session_state["text"] = pasted_text
        else:
            st.warning("Please paste some text first.")

st.divider()

# ── Analysis Pipeline ─────────────────────────────────────
if st.session_state.get("run"):
    st.session_state["run"] = False

    with st.spinner("🔄 Running analysis pipeline..."):

        # Step 1: Extract text
        progress = st.progress(0, text="Extracting text...")
        if st.session_state["type"] == "image":
            raw_text = extract_text_from_image(st.session_state["path"])
        elif st.session_state["type"] == "pdf":
            raw_text = extract_text_from_pdf(st.session_state["path"])
        else:
            raw_text = st.session_state.get("text", "")

        if not raw_text.strip():
            st.error("❌ Could not extract text. Try a clearer image or paste the text directly.")
            st.stop()

        progress.progress(25, text="Detecting language...")
        lang = detect_language(raw_text)

        # Step 2: NLP Analysis
        progress.progress(50, text="Running NLP analysis...")
        analysis = analyze_text(raw_text)

        # Step 3: AI Summary
        progress.progress(75, text="Generating AI summary...")
        summary = generate_summary(raw_text, analysis, lang, api_key=openai_key if openai_key else None)

        # Step 4: Map
        progress.progress(90, text="Building crime hotspot map...")
        locations = [ent["text"] for ent in analysis["entities"] if ent["label"] in ("GPE", "LOC", "FAC")]
        crime_map = build_map(locations, analysis)

        progress.progress(100, text="Done!")

    # ── Results ────────────────────────────────────────────
    st.success("✅ Analysis complete!")

    tab1, tab2, tab3, tab4 = st.tabs(["📝 Extracted Text", "🧠 NLP Analysis", "🗺️ Crime Map", "📊 AI Summary"])

    with tab1:
        lang_label = {"en": "🇬🇧 English", "he": "🇮🇱 Hebrew", "ar": "🇸🇦 Arabic"}.get(lang, "🌍 Unknown")
        st.markdown(f"**Detected Language:** {lang_label}")
        st.text_area("Extracted Text", raw_text, height=300)

    with tab2:
        col_a, col_b, col_c = st.columns(3)
        sentiment = analysis.get("sentiment", {})
        label = sentiment.get("label", "N/A")
        score = sentiment.get("score", 0)
        emoji = "😡" if "negative" in label.lower() else "😊" if "positive" in label.lower() else "😐"

        col_a.metric("Sentiment", f"{emoji} {label.capitalize()}")
        col_b.metric("Confidence", f"{score:.1%}")
        col_c.metric("Entities Found", len(analysis.get("entities", [])))

        st.markdown("---")
        st.subheader("Named Entities")

        entities = analysis.get("entities", [])
        if entities:
            from collections import defaultdict
            grouped = defaultdict(list)
            label_map = {
                "PERSON": "👤 People",
                "ORG": "🏛️ Organizations",
                "GPE": "📍 Locations",
                "LOC": "🌍 Geographic",
                "DATE": "📅 Dates",
                "EVENT": "⚡ Events",
                "FAC": "🏢 Facilities",
            }
            for ent in entities:
                grouped[ent["label"]].append(ent["text"])

            for label_key, items in grouped.items():
                display = label_map.get(label_key, label_key)
                unique_items = list(set(items))
                st.markdown(f"**{display}:** {' · '.join([f'`{i}`' for i in unique_items])}")
        else:
            st.info("No named entities detected.")

        st.markdown("---")
        st.subheader("Keywords")
        keywords = analysis.get("keywords", [])
        if keywords:
            st.markdown(" ".join([f"`{k}`" for k in keywords]))

    with tab3:
        if crime_map:
            from streamlit.components.v1 import html
            html(crime_map, height=500)
        else:
            st.info("🗺️ No mappable locations found in this article. Try an article with city/country names.")

    with tab4:
        st.subheader("AI-Generated Summary Report")
        st.markdown(summary)
        st.download_button(
            "⬇️ Download Summary",
            summary,
            file_name="crime_news_summary.txt",
            mime="text/plain"
        )
