# 🔍 AI Crime News Analyzer

> **Multimodal AI pipeline** that extracts text from news article screenshots or PDFs, performs multilingual NLP analysis (NER + sentiment), plots crime hotspots on an interactive map, and generates an AI-powered intelligence report.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red?logo=streamlit)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-green?logo=openai)
![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow)
![spaCy](https://img.shields.io/badge/spaCy-3.7-blue)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## 🎯 What It Does

1. **Upload** a news article screenshot (PNG/JPG) or PDF
2. **OCR** extracts text using OpenCV + Tesseract — supports English, Hebrew & Arabic
3. **NLP pipeline** runs Named Entity Recognition, sentiment analysis, and keyword extraction
4. **Interactive map** geocodes all locations and plots a crime hotspot heatmap
5. **AI report** uses GPT-4o to generate a structured intelligence report

---

## ✨ Key Features

- 🌍 **Multilingual** — English, Hebrew (עברית), Arabic (العربية)
- 🖼️ **Multimodal** — processes both images and PDFs
- 🗺️ **Interactive crime map** with heatmap overlay (Folium)
- 🧠 **Named Entity Recognition** — people, organizations, locations, dates
- 😡 **Sentiment Analysis** — multilingual XLM-RoBERTa model
- 📊 **AI Summary** — structured intelligence report via GPT-4o
- ✍️ **Text paste mode** — no image needed, paste article directly

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| OCR | OpenCV + Tesseract (multilingual) |
| NLP / NER | HuggingFace `bert-base-multilingual-cased-ner-hrl` + spaCy |
| Sentiment | `cardiffnlp/twitter-xlm-roberta-base-sentiment` |
| AI Summary | OpenAI GPT-4o |
| Map | Folium + Geopy |
| UI | Streamlit |
| Deployment | Hugging Face Spaces |

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/bara-luch/crime-news-analyzer.git
cd crime-news-analyzer
```

### 2. Run the auto-installer (handles everything)
```bash
python setup.py
```
This installs all Python packages, downloads the spaCy model, installs Tesseract OCR, and creates your `.env` file automatically.

### 3. (Optional) Add your OpenAI API key
```bash
# Edit .env and paste your key — app works without it too
OPENAI_API_KEY=your_key_here
```

### 4. Launch the app
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

> **Note:** First run downloads ~1GB of AI models. This only happens once.

---

## 📁 Project Structure

```
crime-news-analyzer/
│
├── app.py                    # Streamlit main application
│
├── ocr/
│   └── extractor.py          # OCR pipeline (OpenCV + Tesseract)
│
├── nlp/
│   ├── analyzer.py           # NER + sentiment + keyword extraction
│   └── summarizer.py         # OpenAI GPT-4o summary generation
│
├── map/
│   └── visualizer.py         # Folium interactive map builder
│
├── utils/
│   └── language.py           # Language detection (EN/HE/AR)
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🌐 Multilingual Support

This project natively supports three languages, leveraging my background as a native Hebrew and Arabic speaker:

| Language | OCR | NER | Sentiment | Notes |
|---|---|---|---|---|
| English | ✅ | ✅ | ✅ | Full support |
| Hebrew | ✅ | ✅ | ✅ | RTL text handling |
| Arabic | ✅ | ✅ | ✅ | RTL text handling |

---

## 📊 Example Output

**Input:** BBC News article about a crime incident  
**Output:**
- Sentiment: `Negative` (94.2% confidence)
- Entities: 3 people, 2 organizations, 4 locations, 2 dates
- Crime types: `violence`, `arrest`
- Map: 4 locations geocoded and plotted with heatmap
- AI Report: structured 6-section intelligence briefing

---

## 🔮 Future Improvements

- [ ] Real-time news scraping (RSS feed integration)
- [ ] Time-series crime trend analysis across multiple articles
- [ ] Export reports as PDF
- [ ] Multi-article comparison dashboard
- [ ] Fine-tuned crime-specific NER model

---

## 👤 Author

**Bara Luch** — ML Engineer & Data Scientist  
📍 Tel Aviv, Israel  
🔗 [LinkedIn](https://linkedin.com/in/bara-luch) · 💻 [GitHub](https://github.com/bara-luch)

---

## 📄 License

MIT License — free to use, modify, and distribute.
