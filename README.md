# Fake News Detection with TF-IDF

Live app: **[tfidffakenewsdetection.streamlit.app](https://tfidffakenewsdetection.streamlit.app/)**

A text classifier that labels a news article as fake or real, built on TF-IDF vectorization and logistic regression, served through a small [Streamlit](https://streamlit.io/) app.

## Data

`Fake.csv` and `True.csv` supply the articles, labeled `0` and `1`. Only the `text` column is kept; `title`, `subject`, and `date` are dropped. Text is lowercased and stripped of brackets, HTML, URLs, punctuation, and any word containing a digit before vectorizing.

## Model

`TfidfVectorizer` converts the cleaned text to features, and a `LogisticRegression` classifier is fit on a 75/25 train-test split (`random_state=42`). Both are saved with `joblib` as `vectorizer.jb` and `model.jb`, which `app.py` loads to serve predictions.

## Results

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| 0 (fake) | 0.99 | 0.99 | 0.99 | 5895 |
| 1 (real) | 0.98 | 0.99 | 0.99 | 5330 |
| **Accuracy** | | | **0.99** | 11225 |


## Files

| File | What it holds |
|---|---|
| `app.ipynb` | Data loading, cleaning, training, evaluation |
| `app.py` | Streamlit app |
| `model.jb`, `vectorizer.jb` | Saved model and vectorizer |
