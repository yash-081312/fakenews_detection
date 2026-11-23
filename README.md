# Fake News Detection & News Circulation Analyzer
### Data Science Essentials (DSE) Capstone Project

**Student Name:** Yaswanth  
**Enrollment Number:** 92400120483  
**Branch / Department:** Computer Science and Engineering (CSE), Department of Computer Engineering  
**Faculty / University:** Faculty of Engineering and Technology, Marwadi University, Rajkot, Gujarat, India  
**Dataset Source:** [Kaggle - Fake News Detection by Vishakh Dapat](https://www.kaggle.com/datasets/vishakhdapat/fake-news-detection)  

---

## 📌 1. Project Overview & New Capabilities

This project delivers an interactive, production-grade Machine Learning and Natural Language Processing (NLP) web platform for **Fake News Detection and News Circulation Analysis**.

### 🌟 Key Features:
1. **Interactive Web UI (Streamlit):** Clean, graphical user interface with real-time gauges, cards, and interactive controls.
2. **Multi-Source Link Analysis:**
   - 📖 **Wikipedia Pages:** Automatically extracts article summary and body paragraphs via official APIs and web scraping.
   - 🎥 **YouTube Videos:** Extracts video titles, descriptions, and auto-transcribes spoken video speech/captions using `youtube-transcript-api`.
   - 🌐 **Web & News Links:** Scrapes and parses articles from any public news URL (e.g., Reuters, BBC, blogs).
   - ✍️ **Direct Text / Headlines:** Paste custom text or viral posts.
3. **Deep-Dive Circulation & Deception Scoring:**
   - **Fake / Misinformation Risk Percentage (%)**
   - **Real Authenticity / Credibility Percentage (%)**
   - **Confidence Level (%)**
   - **What News is Being Circulated:** Core claim extraction, circulating entities/topics (people, agencies, events), and domain context.
   - **Forensic Linguistic Breakdown:** Sensationalist clickbait red flags vs credible journalistic anchors, all-caps emphasis, and punctuation intensity.
4. **Kaggle Dataset Explorer:** Search, filter, and inspect trending stories circulated in the 9,900-article benchmark corpus.
5. **Slide 9 Evaluation Dashboard:** Interactive view of all 5 benchmark ML models with faculty viva defense notes.

---

## 📊 2. Dataset Summary

- **Source:** Kaggle (`vishakhdapat/fake-news-detection`)
- **Total Articles:** 9,900 raw entries
- **Deduplication:** 35 exact duplicate records removed $\rightarrow$ **9,865 unique samples**
- **Class Balance:**
  - **Fake News:** 5,000 articles (50.68%)
  - **Real News:** 4,865 articles (49.32%)
- **Data Attributes:**
  - `Text`: Raw article title and body content.
  - `label`: Binary classification target (`Fake` vs `Real`).

---

## 🏆 3. Experimental Results (Test Set - 1,973 samples)

| Algorithm | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC (%) | Train Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear SVM** *(Top Model)* | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **0.76 s** |
| **Random Forest** | **99.95%** | **100.00%** | **99.90%** | **99.95%** | **100.00%** | **1.79 s** |
| **Logistic Regression** | **99.90%** | **99.80%** | **100.00%** | **99.90%** | **100.00%** | **0.23 s** |
| **Gradient Boosting** | **99.85%** | **99.90%** | **99.80%** | **99.85%** | **99.90%** | **68.23 s** |
| **Multinomial Naive Bayes** | **98.58%** | **98.60%** | **98.60%** | **98.60%** | **99.94%** | **0.02 s** |

---

## 🚀 4. How to Launch the Web UI

### Option A: Double-Click (Windows)
Double-click `run_ui.bat` in the project root.

### Option B: Python Command
```powershell
python app.py
```
*(Or run `streamlit run streamlit_app.py`)*

The web application will open automatically in your browser at `http://localhost:8501`.

---

## 📁 5. Repository Structure

```
dse_project/
│
├── streamlit_app.py                          # Full-featured Streamlit Web Application
├── run_ui.bat                                # Windows 1-click UI launcher
├── app.py                                    # Unified launcher (Web UI, CLI, text mode)
├── requirements.txt                          # Dependencies
├── README.md                                 # Project documentation
│
├── src/
│   ├── link_extractor.py                     # Extracts content from Wikipedia, YouTube, Web URLs
│   ├── news_analyzer.py                      # Circulation analysis, Fake %, Real %, XAI signals
│   ├── preprocessor.py                       # NLP cleaning, stopword removal, lemmatization
│   ├── data_loader.py                        # Dataset ingestion & train/test splitting
│   ├── train.py                              # ML benchmarking & figure generation
│   └── predict.py                            # Core inference engine
│
├── data/
│   ├── raw/fake_and_real_news.csv            # Original Kaggle dataset
│   └── processed/cleaned_news.csv            # Preprocessed & deduplicated corpus
│
├── models/
│   ├── best_model.pkl                        # Serialized top model (Linear SVM)
│   ├── tfidf_vectorizer.pkl                  # Fitted TF-IDF feature extractor
│   └── [logistic_regression, linear_svm, etc].pkl
│
├── notebooks/
│   └── fake_news_detection_eda_and_modeling.ipynb # Complete step-by-step Jupyter Notebook
│
├── results/
│   ├── metrics_summary.csv                   # CSV of all evaluation metrics
│   ├── classification_reports.txt            # Precision/recall per class
│   └── figures/                              # Confusion matrices, ROC curves, Word clouds
│
└── presentation/
    ├── DSE_Presentation_Slide_By_Slide_Content.md  # 12-slide copy-paste script for Canva/PPT
    └── slide_9_metrics_table.md                    # Slide 9 viva defense notes
```

---

## 🎤 6. Presentation Resources

- **Slide Deck Script:** [presentation/DSE_Presentation_Slide_By_Slide_Content.md](file:///c:/Users/onepi/OneDrive/Documents/dse_project/presentation/DSE_Presentation_Slide_By_Slide_Content.md)
- **Slide 9 Viva Defense:** [presentation/slide_9_metrics_table.md](file:///c:/Users/onepi/OneDrive/Documents/dse_project/presentation/slide_9_metrics_table.md)
