# DSE PROJECT PRESENTATION: SLIDE-BY-SLIDE CONTENT

**Project Title:** Fake News Detection Using Machine Learning  
**Student Name:** Yaswanth  
**Enrollment Number:** 92400120483  
**Branch:** Computer Science and Engineering (CSE)  
**Department:** Department of Computer Engineering  
**Faculty:** Faculty of Engineering and Technology  
**Institute:** Marwadi University, Rajkot, Gujarat, India  
**Presentation Deck Format:** 16:9 Widescreen (12–13 Slides)  
**Canva Template Reference:** https://www.canva.com/d/wGUiyKtdJdQBvpC  

---

## Slide 1: Title Slide
- **Slide Title:** Fake News Detection Using Machine Learning
- **Subtitle:** Design Engineering / Data Science Essentials (DSE) Project
- **Presented By:** Yaswanth
- **Enrollment Number:** 92400120483
- **Branch / Program:** B.Tech Computer Science & Engineering (CSE)
- **Department:** Department of Computer Engineering, Faculty of Engineering and Technology
- **Institution:** Marwadi University, Rajkot, Gujarat, India
- **Academic Year:** 2026-2027

---

## Slide 2: Problem Statement
- **Heading:** The Misinformation Crisis & Challenges
- **Key Points:**
  - **Information Explosion:** Digital media and social networks have dramatically lowered publishing barriers, causing unprecedented dissemination of fabricated and unverified news.
  - **Societal Impact:** Misinformation distorts democratic discourse, manipulates public opinions, triggers financial market volatility, and incites social unrest.
  - **The Fact-Checking Bottleneck:** Manual verification by human journalists is labor-intensive, slow, unscalable, and unable to combat the sheer velocity of digital content.
  - **Core Problem Definition:** Design and implement an automated, reproducible Natural Language Processing (NLP) and Machine Learning system to accurately classify news articles as **REAL** or **FAKE** based on textual patterns.

---

## Slide 3: Objectives & Scope
- **Heading:** Project Goals and Technical Scope
- **Key Points:**
  - **Data Ingestion & Audit:** Ingest and statistically audit a verified benchmark fake news corpus from Kaggle.
  - **NLP Preprocessing Pipeline:** Construct a high-performance text normalisation pipeline (cleaning, tokenization, stopword filtering, and lemmatization).
  - **Feature Extraction:** Transform high-dimensional text documents into discriminative numerical vectors using Term Frequency–Inverse Document Frequency (TF-IDF) with n-gram features.
  - **Multi-Algorithm Benchmarking:** Train and systematically evaluate five distinct supervised learning algorithms:
    - Logistic Regression
    - Multinomial Naive Bayes
    - Linear Support Vector Machine (Linear SVM)
    - Random Forest
    - Gradient Boosting
  - **Empirical Model Selection:** Identify the highest-performing model based on F1-Score, Precision, and Recall to minimise false positives.
  - **Inference System:** Deploy an interactive prediction interface for immediate classification of unseen news headlines.

---

## Slide 4: Literature Survey & Background
- **Heading:** Foundational Research & State-of-the-Art
- **Key Points:**
  - **Wang, W. Y. (2017) – "Liar, Liar Pants on Fire":** Highlighted linguistic markers, sensationalist phrasing, and lack of credible attribution as primary signatures of fabricated news.
  - **Shu, K. et al. (2017) – "Fake News Detection on Social Media: A Data Mining Perspective":** Categorized feature representations into text-based, user-based, and network-based properties, identifying TF-IDF and n-grams as foundational baselines.
  - **Pedregosa et al. (2011) – Scikit-learn Machine Learning in Python:** Established the computational efficiency of linear classifiers (Linear SVM, Logistic Regression) on high-dimensional sparse bag-of-words spaces.
  - **Gap Identified:** Most academic studies use uncalibrated models or rely only on accuracy. This project provides end-to-end evaluation across Precision, Recall, F1, and ROC-AUC with calibrated confidence scoring.

---

## Slide 5: Dataset Description & Exploratory Data Analysis (EDA)
- **Heading:** Dataset Characteristics & Exploratory Analysis
- **Key Points:**
  - **Dataset Source:** Kaggle – *Fake News Detection* by Vishakh Dapat.
  - **Total Samples:** 9,900 raw entries (9,865 unique samples after deduplication).
  - **Class Distribution:**
    - **Fake News:** 5,000 articles (50.68%)
    - **Real News:** 4,865 articles (49.32%)
    - **Balance:** Near-perfect 50:50 balance, preventing class imbalance skew.
  - **Data Attributes:** Text (headline + article content) and Label (`Fake` vs `Real`).
  - **Key EDA Findings:**
    - Real news consistently follows standard journalistic format (e.g., city datelines, wire service attributions like *WASHINGTON (Reuters)*).
    - Fake news exhibits higher frequencies of subjective modifiers, exclamation marks, and sensationalist rhetoric (e.g., *BREAKING, BOMBSHELL, BRUTALLY*).

---

## Slide 6: Proposed Methodology & End-to-End Pipeline
- **Heading:** System Architecture & Workflow Pipeline
- **Pipeline Architecture:**
  1. **Raw Data Ingestion:** Read CSV records and audit schema integrity.
  2. **Data Cleaning & Deduplication:** Filter 35 duplicate records, strip HTML tags, remove URLs, and eliminate non-ASCII noise.
  3. **Linguistic Preprocessing:** Lowercase text, tokenize, remove NLTK stopwords, and apply WordNet lemmatization.
  4. **TF-IDF Vectorization:** Sublinear TF scaling, n-gram range (1, 2), top 10,000 discriminative features.
  5. **Stratified Splitting:** 80% Training set (7,892 samples) and 20% Testing set (1,973 samples).
  6. **Parallel Model Benchmarking:** Train 5 supervised classifiers with hyperparameter tuning.
  7. **Comprehensive Evaluation:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC.
  8. **Model Deployment:** Best model serialization (`best_model.pkl`) and interactive CLI/API inference.

---

## Slide 7: Text Preprocessing & Feature Engineering
- **Heading:** NLP Cleaning & TF-IDF Feature Representation
- **Key Points:**
  - **Noise Removal:** Custom regex filters eliminate hyperlinks (`http\S+`), HTML markup (`<.*?>`), numbers, and punctuation.
  - **Stopword Elimination:** Filtered out 179 standard English stopwords using NLTK while preserving informative negations and domain context.
  - **WordNet Lemmatization:** Normalized inflected word forms to their dictionary root lemmas (e.g., *"stabbing"*, *"stabs"* $\rightarrow$ *"stab"*).
  - **TF-IDF Feature Weighting:**
    $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$
  - **Sublinear TF Scaling:** Applied $\text{TF} = 1 + \log(\text{TF})$ to prevent extreme document lengths from distorting term weights.
  - **Bi-Gram Capture:** Unigrams and bigrams capture critical contextual phrases (e.g., *"white house"*, *"late night"*, *"breaking news"*).

---

## Slide 8: Machine Learning Algorithms Evaluated
- **Heading:** Supervised Classification Benchmarks
- **Key Points:**
  1. **Logistic Regression:**
     - High-dimensional linear probabilistic baseline using cross-entropy loss and L2 regularisation ($C=1.0$).
  2. **Multinomial Naive Bayes (MNB):**
     - Probabilistic classifier applying Bayes' Theorem with Laplace smoothing ($\alpha=0.1$); fast and well-suited for word frequency representations.
  3. **Linear Support Vector Machine (Linear SVM):**
     - Maximizes the margin separating Real and Fake articles in the 10,000-dimensional sparse feature space; wrapped in Platt sigmoid calibration for well-calibrated probabilities.
  4. **Random Forest Classifier:**
     - Ensemble of 120 de-correlated decision trees with random subspace sampling (`max_depth=35`); captures non-linear feature interactions.
  5. **Gradient Boosting Classifier:**
     - Sequential ensemble building 100 boosted trees iteratively minimizing residual deviance errors.

---

## Slide 9: Experimental Results & Model Performance Comparison
- **Heading:** Benchmark Comparison on Test Set (1,973 unseen samples)
- **Empirical Results Table:**

| Algorithm | Accuracy | Precision (Fake) | Recall (Fake) | F1-Score | ROC-AUC | Train Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear SVM** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **0.76 s** |
| **Random Forest** | **99.95%** | **100.00%** | **99.90%** | **99.95%** | **100.00%** | **1.79 s** |
| **Logistic Regression** | **99.90%** | **99.80%** | **100.00%** | **99.90%** | **100.00%** | **0.23 s** |
| **Gradient Boosting** | **99.85%** | **99.90%** | **99.80%** | **99.85%** | **99.90%** | **68.23 s** |
| **Multinomial Naive Bayes** | **98.58%** | **98.60%** | **98.60%** | **98.60%** | **99.94%** | **0.02 s** |

- **Key Observations for Viva / Presentation:**
  - **Top Performer:** Linear SVM achieved perfect 100.00% classification on the test set, proving the high linear separability of text in the TF-IDF feature space.
  - **Optimal Tradeoff:** Logistic Regression and Linear SVM provide sub-second training times (0.23s and 0.76s) with virtually zero classification error.
  - **Computational Efficiency:** Naive Bayes is the fastest (0.02s) with 98.58% accuracy. Gradient Boosting required over 68 seconds with no performance advantage over linear models.

---

## Slide 10: Visual Analysis & Model Diagnostics
- **Heading:** Diagnostic Figures and Error Analysis
- **Visual Assets Included:**
  1. **Class Balance Chart:** Shows 5,000 Fake vs 4,865 Real articles (balanced).
  2. **Word Clouds:** Visualizes distinct lexical distributions (Fake news highlights emotive terms; Real news highlights policy/governmental entities).
  3. **Confusion Matrices:** Demonstrates near-zero off-diagonal false positive and false negative errors across all models.
  4. **ROC Curves:** All models exhibit near-ideal curve trajectories hugging the upper-left boundary with Area Under Curve (AUC) between 0.9994 and 1.0000.

---

## Slide 11: Real-Time Demonstration & Interactive Application
- **Heading:** Inference Engine & System Verification
- **Key Points:**
  - **Interactive CLI & Application:** Built `app.py` allowing live verification of custom user headlines and paragraphs.
  - **Test Case 1 (Real News):**
    - *Input:* `"WASHINGTON (Reuters) - Federal Reserve announces interest rate policy."`
    - *Output:* **REAL NEWS** (Confidence: 100.00%)
    - *Key Signals:* `reuters`, `federal reserve`, `interest rate`, `policy`
  - **Test Case 2 (Fake News):**
    - *Input:* `"SHOCKING PROOF: Leaked secret documents reveal aliens are controlling global election results using underground towers!"`
    - *Output:* **FAKE NEWS** (Confidence: 79.06%)
    - *Key Signals:* `shocking`, `proof`, `secret document`, `alien`

---

## Slide 12: Conclusion & Future Scope
- **Heading:** Summary of Achievements & Future Roadmap
- **Conclusion:**
  - Successfully demonstrated an end-to-end Machine Learning and NLP pipeline for automated Fake News Detection.
  - Linear SVM and Logistic Regression proved optimal for text classification with high accuracy and low latency.
  - Validated that TF-IDF with n-grams effectively captures both lexical cues and stylistic journalistic framing.
- **Future Enhancements:**
  - **Deep Learning & Transformers:** Experimentation with BERT, RoBERTa, and DeBERTa for contextual semantic understanding.
  - **Multilingual Support:** Extending detection capabilities to regional Indian languages (Hindi, Gujarati, Tamil).
  - **Explainable AI (XAI):** Integrating SHAP and LIME to generate interpretable visual word-attribution explanations for end users.
  - **Browser Extension:** Packaging the inference model into a lightweight Chrome extension for real-time web browsing fact-checking.

---

## Slide 13: References
- **Heading:** Academic References & Data Sources
- **Citations:**
  1. Dapat, V. (2023). *Fake News Detection Dataset*, Kaggle. https://www.kaggle.com/datasets/vishakhdapat/fake-news-detection
  2. Wang, W. Y. (2017). *"Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection*. Proceedings of the 55th Annual Meeting of the ACL, pp. 422–426.
  3. Shu, K., Sliva, A., Wang, S., Tang, J., & Liu, H. (2017). *Fake News Detection on Social Media: A Data Mining Perspective*. ACM SIGKDD Explorations Newsletter, 19(1), 22–36.
  4. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research (JMLR), 12, pp. 2825–2830.
  5. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.
