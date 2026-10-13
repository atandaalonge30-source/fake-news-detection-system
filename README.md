# Fake News Detection System Using Natural Language Processing

This is a complete HND project implementation matching the supplied proposal: it treats fake-news detection as a **binary text-classification** problem. It trains a TF-IDF (unigram + bigram) and Logistic Regression model, uses train/validation/test splits, saves the trained pipeline, creates evaluation reports, and provides a Flask web demonstration.

> Important: a model prediction is not proof that a story is false or true. It detects patterns learned from its training data. For real use, combine it with source checks, dates, primary evidence and professional fact-checking.

## 1. Prerequisites

Install Python 3.10 or newer. Open PowerShell in this folder and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass`, then activate again. On macOS/Linux use `source .venv/bin/activate`.

## 2. Obtain and prepare data

Use a legally accessible labelled fake/real news dataset. A common student option is the Kaggle **Fake and Real News Dataset**, which has `Fake.csv` and `True.csv` files with `title` and `text` columns. Download it yourself, then place the files in `data/`:

```
data/Fake.csv
data/True.csv
```

Do **not** evaluate only on the four-row `sample_news.csv`; it exists solely to show the expected format. Your data should be reasonably balanced and contain enough records for all splits. Remove blank, duplicate and irrelevant records; keep the source and collection date in a separate data log for your project report.

If your dataset is one CSV instead, it must contain `label` plus `text` (or `title` + `text`) columns. `label` may be `fake`/`real` or `0`/`1` (`0 = fake`, `1 = real`). A CSV with a `target` column is also supported for the linked Fake and Real News Dataset (`0 = real`, `1 = fake`).

For this project, you can use the educational Fake and Real News Dataset mirrored on Hugging Face: [dataset page](https://huggingface.co/datasets/haoxianc/gilchr_fake-and-real-news-dataset). Its dataset card identifies an MIT license and credits Gilchrist; keep the attribution and license when redistributing the data or a model trained from it. Dataset labels and quality have limitations, so the result is a demonstration classifier, not a fact-checker.

To train with that dataset after downloading `fake_real_news.csv` into `data/`, run:

```powershell
python -m src.train --data data/fake_real_news.csv
```

## 3. Train and evaluate

From the project root, for the two-file dataset run:

```powershell
python -m src.train --fake-csv data/Fake.csv --true-csv data/True.csv
```

Or, for a single labelled CSV:

```powershell
python -m src.train --data data/my_labelled_news.csv
```

The script does the following reproducibly (random seed 42):

1. Combines title and article body, removes blank/very short items and standardizes labels.
2. Stratifies the data into 70% training, 15% validation and 15% unseen test data.
3. Normalizes text: lowercases it, removes URLs/HTML/punctuation and collapses extra spaces.
4. Extracts TF-IDF unigram and bigram features, excluding English stop words.
5. Fits class-balanced Logistic Regression on the training portion.
6. Reports validation and test accuracy, precision, recall and F1; then saves the fitted pipeline.

Outputs go to `artifacts/`:

- `fake_news_model.joblib` — trained model for reuse.
- `metrics.json` — numeric results for Chapter Four.
- `test_predictions.csv` — actual vs predicted records for error analysis.
- `confusion_matrix.png` — visual evaluation result.

## 4. Test a single item from the terminal

```powershell
python -m src.predict --text "Paste a news headline or full article here."
```

## 5. Run the web system

After training:

```powershell
python app.py
```

Open `http://127.0.0.1:5000` in a browser. Paste text and select **Analyse text**. Stop the server with `Ctrl+C`.

## 6. Publish the web app

Render is the simplest option for this Flask application. Vercel is primarily
designed for frontend and serverless applications; this app needs a persistent
Python web process and a trained scikit-learn model, so Render is the more
straightforward fit.

1. Download a labelled dataset and train the model using one of the commands in
   section 3. Confirm that `artifacts/fake_news_model.joblib` was created.
2. Put this project folder in a GitHub repository. The trained model is
   git-ignored along with generated reports, so it must be explicitly added:

   ```powershell
   git add -f artifacts/fake_news_model.joblib
   git add app.py requirements.txt README.md src
   git commit -m "Prepare app for deployment"
   git push
   ```

   Only deploy a model trained on data you are allowed to redistribute. If the
   model is too large for GitHub, store it in an external model store and set
   Render's `MODEL_PATH` environment variable to a downloaded model path.
3. In Render, create a **New + → Web Service**, connect the GitHub repository,
   choose Python, and set:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `gunicorn --bind 0.0.0.0:$PORT app:app`
4. Deploy. Render gives the service a public `*.onrender.com` URL. Open that
   URL to use the app; `/health` should return `{"status":"ok"}` when the model
   is present.

The app cannot make meaningful predictions until it has been trained on a
labelled dataset. The included sample CSV is only a format example and is too
small for training. Do not treat predictions or their confidence as proof that
a story is true or false.

## 7. Run basic automated tests

```powershell
pytest -q
```

## Suggested Chapter Four report structure

1. State dataset name, source, total cleaned rows and class distribution.
2. Explain the 70/15/15 split and why the test set was not used to tune the model.
3. Insert `artifacts/confusion_matrix.png` and a table built from `artifacts/metrics.json`.
4. Discuss precision (how many predicted labels were correct), recall (how many actual labels were found), F1 (their balance) and errors seen in `test_predictions.csv`.
5. State limitations: dataset bias, changing writing styles, labels may be imperfect, and the system is not fact verification.

## Project layout

```
app.py                    Flask user interface
src/text_processing.py    NLP normalization
src/train.py              data loading, model training and evaluation
src/predict.py            command-line prediction
tests/                    basic unit tests
data/                     place downloaded data here
artifacts/                created model and evaluation outputs
```
