# 🧠 Fake-News • 💳 Credit-Card-Fraud • 🎬 Movies Recommendation (End-to-End ML Suite)

Three production-style ML mini-apps with **training notebooks/scripts** + **Streamlit UIs**:

1) **Fake News Detection** — TF-IDF (1–2) + Logistic Regression, URL smart-scrape, token-level explanations  
2) **Credit Card Fraud** — Supervised LR (balanced) + **PCA anomaly** + **decision fusion** (OR/AND/AVG)  
3) **Movie Recommender** — Fast hybrid CF (UserCF ⨉ ItemCF), filters, diversity (MMR), TMDb/OMDb optional

> Designed for local runs (Windows/WSL/macOS/Linux). All code is schema-tolerant, feature-order safe, and saves models to `models/`.

---

## 📚 Datasets

- **Fake News:** https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets  
  (expects `True.csv` + `Fake.csv`)
- **Credit Card Fraud:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  
  (expects `creditcard.csv`)
- **Movies:** https://www.kaggle.com/datasets/parasharmanas/movie-recommendation-system  
  (expects MovieLens-style `ratings.csv`, `movies.csv`)

Put files under `data/` as shown below (or edit paths in scripts).

---

## 📁 Repo Layout (suggested)

.
├─ fake_news/
│ ├─ train_fake_news.ipynb / .py
│ ├─ app_fake_news_streamlit.py
│ └─ fake_news_tfidf_lr.joblib # created after training
├─ fraud/
│ ├─ train_fraud.ipynb / .py
│ ├─ fraud_lr_balanced.joblib # created after training
│ ├─ fraud_pca_anomaly.joblib # created after training
│ └─ app_fraud_streamlit.py
├─ movies/
│ ├─ train_movies.ipynb / .py
│ ├─ movie_recommender.joblib # created after training
│ └─ app_movie_streamlit.py
├─ data/
│ ├─ News/True.csv, News/Fake.csv
│ ├─ creditcard/creditcard.csv
│ └─ Movie/ratings.csv, Movie/movies.csv
├─ models/ # auto-created
├─ outputs/ # auto-created
└─ README.md

yaml
Copy code

> You can keep your existing single-folder layout; filenames in this README match the code you provided.

---

## ⚙️ Environment

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\activate

pip install -U pip
pip install -r requirements.txt
requirements.txt

bash
Copy code
numpy
pandas
scikit-learn
matplotlib
joblib
scipy
streamlit
beautifulsoup4
lxml
requests
newspaper3k
polars; python_version>="3.10"  # optional speed-up for movies trainer
🧩 Common Utility
All trainers include a robust loader:

python
Copy code
def SAFE_READ_CSV(preferred_paths, fallback_msg):
    # Tries multiple paths; falls back to prompting a manual path.
Place files in data/... or provide the absolute path when prompted.

📰 Project 1 — Fake News Detection
Training script core (you provided):

Loads True.csv + Fake.csv

Unifies to a text column

Pipeline = TfidfVectorizer(1–2, min_df=3, max_df=0.9, stop_words='english') → LogisticRegression(class_weight='balanced', max_iter=2000)

Saves fake_news_tfidf_lr.joblib

Helper: explain_text() for top tokens pushing FAKE/REAL

Run training

bash
Copy code
cd fake_news
python train_fake_news.py  # or run the .ipynb
Streamlit app

bash
Copy code
streamlit run app_fake_news_streamlit.py
Features:

Paste text or URL; smart scrape (robots.txt aware) via newspaper3k → bs4 fallback

Threshold slider, token contribution tables + optional inline highlights

Batch CSV scoring + download

Ensure fake_news_tfidf_lr.joblib exists in the same folder.

💳 Project 2 — Credit Card Fraud
Training script core (you provided):

Supervised LR (balanced) with StandardScaler

PR-optimal threshold (max F1 on PR curve)

PCA anomaly (RobustScaler + SVD, recon error → [0,1])

Saves:

fraud_lr_balanced.joblib (pipeline + threshold + feature order)

fraud_pca_anomaly.joblib (components + threshold)

Fusion utilities: OR / AND / weighted AVG

Run training

bash
Copy code
cd fraud
python train_fraud.py  # or run the .ipynb
Streamlit app

bash
Copy code
streamlit run app_fraud_streamlit.py
Features:

Single JSON or batch CSV

Two thresholds (LR & PCA) + fusion selector

Strict feature-order safety when scoring

Put creditcard.csv under data/creditcard/ (or edit the path).

🎬 Project 3 — Movie Recommender
Training script core (you provided):

Fast ingest (optional Polars) → Parquet cache

Build CSR rating matrix; L2-normalize user & item matrices

Hybrid CF (UserCF ⨉ ItemCF) + popularity prior

Quick LOO Recall@10 sanity check

Saves movie_recommender.joblib (maps, popularity, metadata, CSR path)

Run training

bash
Copy code
cd movies
python train_movies.py  # or run the .ipynb
Streamlit app

bash
Copy code
streamlit run app_movie_streamlit.py
Features:

Auto-personalization:

If URL has ?user_id=<id> and exists in data → builds genre profile; else popularity start

Session 👍 likes update genre affinity silently

Filters: include/exclude genres, year range, min ratings

Re-ranking: genre boost, recency boost, serendipity, MMR diversity

Optional posters (TMDb) + metadata (OMDb). Paste keys in sidebar.

Put ratings.csv & movies.csv under data/Movie/.

🚀 Quickstart (one-liners)
bash
Copy code
# Fake News
python fake_news/train_fake_news.py
streamlit run fake_news/app_fake_news_streamlit.py

# Fraud
python fraud/train_fraud.py
streamlit run fraud/app_fraud_streamlit.py

# Movies
python movies/train_movies.py
streamlit run movies/app_movie_streamlit.py
🧪 Outputs
models/ — joblibs & bundles for inference

outputs/ — predictions, labeled data (e.g., task2_clusters.csv etc.)# Fake-News-Credit-Card-Fraud-Movies-prediction-ML-
