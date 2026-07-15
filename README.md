# 📰 Fake News Detection using NLP

![Fake News Detection](./docs/fake-news-detective.png)

## 👋 Introduction

Welcome to the Fake News Detection project as part of the IBM AI Course AI101 for Naanmudhalvan. This project aims to develop a fake news detection system using Natural Language Processing (NLP) techniques. The goal is to build a model that can distinguish between fake and real news articles.

The dataset used for this project can be found on Kaggle: [Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset). It contains a collection of news articles labeled as either "fake" or "real," making it suitable for training and evaluating machine learning models.

## 🔍 Project Overview

### 🎯 Objectives

- Implement NLP techniques to preprocess and analyze text data.
- Build and evaluate machine learning models for fake news detection.
- Explore the use of LSTM and BERT-based models for improved accuracy.
- Create an ensemble model that combines the strengths of different architectures.
- Provide a robust fake news detection system that can be used as a valuable tool for media literacy.

### 🗂️ Project Structure

The project is organized as follows:

- **Data Exploration:** Explore the dataset to understand its structure and characteristics.
- **Data Preprocessing:** Clean and preprocess the text data, including tokenization and feature engineering.
- **Model Training:** Train TF-IDF-based Naive Bayes, Decision Tree, Passive Aggressive, Random Forest, and Logistic Regression classifiers.
- **Evaluation:** Evaluate the models using accuracy, precision, recall, F1-score, and confusion matrices.
- **Dataset Bias & Robustness Analysis:** Audit the dataset for label-leaking artifacts, retrain without them, and test generalization on out-of-distribution examples (see Known Limitations below).

## ⚠️ Known Limitations

The ~99% accuracy reported by the classifiers is inflated by artifacts specific to this dataset, not genuine fake-news detection ability:

- 81%+ of real articles open with a wire-service dateline (e.g. `WASHINGTON (Reuters) -`) that no fake article has, letting a model spot "is this formatted like a Reuters story" instead of judging truthfulness.
- The subject column is completely disjoint between real and fake articles — a collection-time artifact.
- When tested on hand-written statements outside this dataset (not scraped from the same sources), accuracy drops to near chance level (~50-56%).

The notebook's final section ("Step 8: Dataset Bias & Robustness Analysis") quantifies this, retrains a leakage-free model, and runs the out-of-distribution test — treat this project as a style/genre classifier trained on one 2016-2017 news scrape, not a general-purpose fact-checker.

## 🧩 Browser Extension + Local Analysis API

Because of the limitations above, the interactive tool built on this project is deliberately framed as a style/manipulation-pattern analyzer, not a fact-checker. It combines the classifier's probability with independent, human-verifiable stylometric signals, and every screen carries the disclaimer: "This tool flags stylistic manipulation patterns, not factual accuracy."

### 📁 Folder layout

```
backend/                # FastAPI server (POST /analyze)
  main.py               # API endpoint, CORS, score blending
  model.py              # joblib model loading (stub fallback if artifacts missing)
  signals.py            # interpretable stylometric signals
  explain.py            # highlighted-span attributions (SHAP/LIME placeholder)
  requirements.txt
  artifacts/            # put model.pkl + vectorizer.pkl here
extension/              # Chrome extension (Manifest V3)
  manifest.json         # activeTab + scripting, host permission for localhost:8000
  content.js            # article extraction + on-page highlighting
  popup.html / popup.js # score gauge, signals list, disclaimer
```

### 🚀 Run the backend

```
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Without model artifacts the server runs with a clearly-labeled stub so the whole pipeline is testable. To use the trained model, export it from the notebook and drop the files in `backend/artifacts/`:

```python
import joblib
joblib.dump(model, "backend/artifacts/model.pkl")
joblib.dump(vectorizer, "backend/artifacts/vectorizer.pkl")
```

Check `FAKE_CLASS` in `backend/model.py` matches your label encoding, and pin scikit-learn in `requirements.txt` to the version you trained with.

### 🧪 Load the extension

- Open `chrome://extensions`, enable Developer mode (top right).
- Click Load unpacked and select the `extension/` folder.
- Open any news article, click the extension icon, then Analyze this page.
- Optionally click Highlight patterns on page to mark flagged sentences in-place (amber = sensationalist style, green = sourcing language).

### 🔧 Extending it

- **Real attributions:** `backend/explain.py` documents how to replace the heuristic spans with LogisticRegression coefficient contributions, SHAP, or LIME.
- **New signals:** add a function in `backend/signals.py` returning a Signal; it automatically flows into the score, the API response, and the popup.
- **DistilBERT variant:** swap `predict_proba_fake()` in `backend/model.py` for a transformers pipeline (dependencies are pre-listed, commented out, in `requirements.txt`).

## 🛠️ Dependencies and Tools

This project utilizes the following tools and dependencies:

- **Python 3:** The programming language used for the project.
- **Jupyter Notebook (optional):** An interactive development environment for data analysis and machine learning.
- **Required Python Libraries:** You will need the following Python libraries, which can be installed using pip:
  - pandas for data manipulation.
  - numpy for numerical operations.
  - scikit-learn for machine learning tasks.
  - nltk (Natural Language Toolkit) for natural language processing tasks.

You can install these libraries using the following command:

```
pip install pandas numpy scikit-learn nltk
```

## 🤖 Machine Learning Algorithms Used

The project uses the following machine learning algorithms and techniques:

- **TF-IDF (Term Frequency-Inverse Document Frequency):** Used for text feature extraction to convert text data into numerical form.
- **Multinomial Naive Bayes:** A classification algorithm for text data often used for spam and fake news detection.
- **Logistic Regression:** A classification algorithm for binary and multi-class classification tasks.
- **Random Forest:** An ensemble learning method for classification and regression tasks.
- **Passive Aggressive Classifier:** A type of online learning algorithm for text classification.
- **Decision Tree:** A classification algorithm that uses a tree structure for decision-making.
- **Train-Test Split:** A technique to split the dataset into training and testing sets for model evaluation.
- **Confusion Matrix:** A tool for evaluating classification model performance.
- **Precision, Recall, F1-Score:** Metrics for evaluating the performance of classification models.
- **ROC Curve (Receiver Operating Characteristic):** Used to assess the performance of binary classification models.
- **Stopwords Removal:** A text preprocessing technique to remove common words that do not contribute much information.
- **Lowercasing:** Converting text to lowercase to ensure uniformity.
- **Tokenization:** Breaking text into words or tokens for analysis.

## 🏁 Getting Started

To run this project on your local machine, follow these steps:

1. Clone this repository:

```
git clone https://github.com/vijaisuria/Fake-News-Detective.git
```

2. Install the necessary dependencies:

3. Download the dataset from Kaggle and place it in the project directory.

4. Run the project using your preferred Python IDE or Jupyter Notebook.

## 💡 Usage

You can use this project to understand the process of fake news detection using machine learning. Follow the steps in the Jupyter Notebook to:

- Load and preprocess the dataset.
- Train and evaluate machine learning models.
- Make predictions for fake news detection.

## 🤝 Contribution Guidelines

Contributions to this project are welcome! If you'd like to contribute, please follow these guidelines:

1. Fork the repository and create your branch from main.

2. Make sure your code follows good coding practices and includes proper comments.

3. Test your changes thoroughly to ensure they don't introduce issues.

4. Submit a pull request, describing the changes you've made and their significance.

## 🙏 Acknowledgments

We would like to express our gratitude to Kaggle for providing the dataset used in this project.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

Happy coding! 🎉 If you have any questions or need assistance, please feel free to reach out to us.

Project maintained by Vijai Suria.
