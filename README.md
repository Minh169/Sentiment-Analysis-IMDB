# Sentiment Analysis on IMDB Reviews

A machine learning project that classifies IMDB movie reviews as positive or negative, comparing classic NLP + ML approaches: Logistic Regression and three flavors of Naive Bayes (Multinomial, Gaussian, Bernoulli).

## Overview

The project investigates how computational models can interpret the emotional tone of user-generated movie reviews. Since platforms like IMDB rely heavily on audience reviews to shape viewing choices and a film's reputation, being able to automatically gauge sentiment offers value to both filmmakers and viewers. The full workflow — from raw text to trained, evaluated models — lives in the notebook `Assessment2_25009034.ipynb`:

1. **Data preparation** — load the IMDB dataset, visualize sentiment distribution and review-length patterns.
2. **Data handling & exploratory analysis** — check for missing values, duplicates, and outliers.
3. **Text preprocessing** — clean and normalize review text (stopword removal, stemming, etc. via NLTK).
4. **Splitting & label encoding** — split into train/test sets and encode sentiment labels.
5. **Model training & evaluation** — vectorize text with TF-IDF, train Logistic Regression and Naive Bayes (Multinomial/Gaussian/Bernoulli) models, and evaluate with accuracy, precision, recall, confusion matrices, and ROC/AUC curves.

## Project structure

```
Sentiment-Analysis-IMDB/
├── Assessment2_25009034.ipynb   # Main notebook: EDA, preprocessing, training, evaluation
├── IMDB Dataset.csv             # Raw IMDB review dataset (review text + sentiment label)
└── dataset/
    ├── imdb_positive.csv        # Positive reviews subset
    └── imdb_negative.csv        # Negative reviews subset
```

## Getting started

### Prerequisites

- Python 3.9+ (or Google Colab — the notebook supports both)
- Jupyter Notebook / JupyterLab, if running locally

### Installation

```bash
git clone https://github.com/Minh169/Sentiment-Analysis-IMDB.git
cd Sentiment-Analysis-IMDB
pip install pandas numpy scikit-learn nltk matplotlib seaborn wordcloud scipy
```

The notebook also downloads the NLTK corpora it needs (stopwords, etc.) on first run.

### Running the notebook

- **Google Colab**: open `Assessment2_25009034.ipynb` in Colab and run the "Mounting Google Drive" cell first, then run the rest top to bottom.
- **Local / Jupyter**: open the notebook with `jupyter notebook Assessment2_25009034.ipynb` and skip the Google Drive mounting cell — use the local-IDE data loading cell instead (the notebook has both paths built in).

## Models compared

| Model | Vectorization |
|---|---|
| Logistic Regression | TF-IDF |
| Multinomial Naive Bayes | TF-IDF |
| Gaussian Naive Bayes | TF-IDF (dense) |
| Bernoulli Naive Bayes | TF-IDF |

Each model is evaluated with accuracy, precision, recall, F1, a confusion matrix, and (for Logistic Regression) an ROC curve, so you can compare which approach handles this dataset best.
