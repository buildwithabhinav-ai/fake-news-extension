# Fake News Detective

A Chrome extension and local API that read the news article you are on and score how **manipulative its writing style** is, with the reasons behind the score.

It flags stylistic patterns such as clickbait wording, sensational phrasing and punctuation habits. It does **not** check facts, and every screen says so: *"This tool flags stylistic manipulation patterns, not factual accuracy."*

## Why it is a style checker, not a fact-checker

A classifier trained on the public Fake/Real News dataset scores about 99% on that dataset's own test split. That number is misleading, and the repo documents why (`IMPROVEMENTS.md`, notebook Step 8):

- 81% of real articles in that dataset open with a wire-service dateline such as `WASHINGTON (Reuters) -`, and the token `(Reuters)` appears in 99.2% of real articles but only 0.04% of fake ones. A model can score well just by spotting "looks like a Reuters story".
- The `subject` column is completely disjoint between real and fake articles, which is an artifact of how the data was collected.
- On hand-written statements from outside the dataset, accuracy drops to near chance (about 50-56%).

So the project retrains without those artifacts (`backend/textclean.py`, `backend/train.py`) and combines the classifier with independent, human-checkable stylometric signals (`backend/signals.py`). It is presented as a writing-style tool trained on one 2016-2017 news scrape.

## How it works

```
extension/   Chrome extension (Manifest V3): extracts the article, shows the score,
             signals and disclaimer, and can highlight flagged sentences on the page
backend/     FastAPI server (POST /analyze)
  main.py      API endpoint, CORS, score blending
  model.py     model loading (clearly labelled stub if artifacts are missing)
  signals.py   interpretable stylometric signals
  textclean.py artifact-free text cleaning
  train.py     trains and compares the original vs cleaned pipeline, exports artifacts
  explain.py   highlighted-span explanations (heuristic; SHAP/LIME is a placeholder)
fakenews_detection.ipynb   exploration and model comparison notebook
```

Models compared: Multinomial Naive Bayes, Logistic Regression, Random Forest, Passive Aggressive and Decision Tree classifiers on TF-IDF features. The served model is TF-IDF with Logistic Regression.

## Run it

You need Python 3 and Chrome.

```bash
./run
```

The first run creates `backend/.venv` and installs dependencies. It starts the API on port 8000 and a landing page on port 8080, and stops both on Ctrl+C. Use `API_PORT=9000 WEB_PORT=9090 ./run` to change ports.

To run the API by hand:

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

To retrain, download the dataset (see Credits), place it in `dataset/`, and run `python3 backend/train.py`.

### Load the extension

1. Open `chrome://extensions` and turn on **Developer mode**.
2. Click **Load unpacked** and choose the `extension/` folder.
3. Open a news article, click the extension icon, then **Analyze this page**.
4. Optionally click **Highlight patterns on page** to mark flagged sentences.

## Credits and licence

This project started from [Fake-News-Detective](https://github.com/vijaisuria/Fake-News-Detective) by **Vijai Suria**, an IBM AI101 course project for Naan Mudhalvan (MIT licence). That project is a Jupyter notebook that compares classical text classifiers on the Kaggle dataset. Steps 1-7 of `fakenews_detection.ipynb` (loading the data, preprocessing, and the classifier comparison), the original README text and the licence come from there.

Added in this repository:

- the Chrome extension (`extension/`)
- the FastAPI analysis service (`backend/`)
- the stylometric signals and score blending
- the dataset artifact audit and artifact-free retraining (`IMPROVEMENTS.md`, notebook Step 8)
- the out-of-distribution test that showed the accuracy drop (notebook Step 8)
- the model export for the API (notebook Step 9)

Dataset: [Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset) on Kaggle. It has its own terms, so download it from Kaggle instead of redistributing it.

Released under the MIT licence. See [LICENSE](LICENSE), which keeps the original copyright notice.

Maintained by [Abhinav](https://github.com/buildwithabhinav-ai).
