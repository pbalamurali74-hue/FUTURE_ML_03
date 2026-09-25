# 🤖 Intent-Based AI Conversational Bot

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![NLP](https://img.shields.io/badge/NLP-Intent%20Classification-009688?style=for-the-badge&logo=openai&logoColor=white)](https://en.wikipedia.org/wiki/Natural_language_processing)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Flask](https://img.shields.io/badge/Flask-Web%20Interface-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![NLTK](https://img.shields.io/badge/NLTK-Text%20Processing-blue?style=for-the-badge)](https://www.nltk.org)

An intelligent, machine-learning-driven conversational assistant designed for automated customer support triage, developed as **Task 3** during the **Future Interns Machine Learning Internship**. Combines structured JSON intent definitions, natural language preprocessing, TF-IDF vectorization, and a trained classification model with a lightweight **Flask** web interface.

---

## 🔍 Problem Statement & Overview

Modern customer service organizations process thousands of routine requests every day. Routing tickets manually creates administrative delays and slows response times for urgent customer issues.

This project delivers an automated conversational agent capable of classifying multi-phrase customer queries into predefined intent categories (e.g., refund requests, billing inquiries, cancellations, technical support, and order status), evaluating confidence scores, and providing immediate contextual responses or graceful fallback routing.

---

## ✨ Key Technical Architecture

1. **Structured JSON Intent Taxonomy (`intents.json`):**
   - Organized schema with intent tags, training input patterns, and randomized conversational responses.
   - Domain-specific coverage for billing, account management, refunds, and support escalations.

2. **NLP Text Preprocessing Pipeline (`scripts/preprocess.py`):**
   - Input cleaning: Lowercasing, punctuation stripping, and tokenization.
   - Lemmatization and stop-word filtering using **NLTK** to normalize morphological query variations.
   - Text vectorization converting user utterances into feature vectors for inference.

3. **Classification & Intent Prediction Engine (`model/intent_classifier.pkl`):**
   - Trained supervised classification algorithm predicting class probability distributions across intent categories.
   - Hardcoded greeting recognition for immediate friendly conversational response.

4. **Confidence Thresholding & Fallback Protection:**
   - Evaluates prediction confidence against defined thresholds.
   - Ambiguous or out-of-domain queries trigger a helpful fallback prompt requesting clarification rather than generating misleading answers.

5. **Interactive Web Application (`chatbot.py`):**
   - Flask REST backend serving static assets and dynamic JSON chat responses.
   - Clean, responsive web UI for real-time live customer interaction.

---

## 📁 Repository Structure

```
FUTURE_ML_03/
├── FUTURE_ML_03-main/
│   ├── chatbot.py           # Flask web application entrypoint & API handler
│   ├── intents.json         # JSON structured intents, patterns, and responses
│   ├── requirements.txt     # Python dependencies (Flask, scikit-learn, NLTK)
│   ├── model/               # Serialized trained model & vectorizer
│   │   ├── intent_classifier.pkl
│   │   └── vectorizer.pkl
│   ├── scripts/             # NLP preprocessing and helper modules
│   │   └── preprocess.py
│   ├── data/                # Training tickets dataset
│   │   └── customer_support_tickets.csv
│   ├── notebooks/           # Exploratory & model training notebooks
│   ├── app/                 # Web interface assets (HTML templates & CSS)
│   └── README.md
└── README.md                # Comprehensive root project documentation
```

---

## 🚀 Running Locally

### 1. Prerequisites
Ensure you have **Python 3.8+** installed.

### 2. Clone Repository
```bash
git clone https://github.com/pbalamurali74-hue/FUTURE_ML_03.git
cd FUTURE_ML_03/FUTURE_ML_03-main
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download NLTK Corpora (if prompted)
```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
```

### 5. Launch the Chatbot Server
```bash
python chatbot.py
```
Open your browser at `http://127.0.0.1:5000` to interact with the conversational assistant.

---

## 👤 Author

**Purushotham Balamurali**  
- **Role:** Machine Learning Intern @ Future Interns  
- **GitHub:** [@pbalamurali74-hue](https://github.com/pbalamurali74-hue)  
- **LinkedIn:** [purushothambalamurali](https://www.linkedin.com/in/purushothambalamurali/)  
- **Portfolio:** [Purushotham Balamurali Portfolio](https://github.com/pbalamurali74-hue)
