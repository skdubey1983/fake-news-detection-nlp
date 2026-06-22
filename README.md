# Fake News Detection using NLP and Machine Learning

## Overview

This project presents an intelligent Fake News Detection System developed using Natural Language Processing (NLP) and Machine Learning techniques. The system automatically classifies textual news content as **Fake News** or **Real News** and provides explainable predictions through confidence analysis, suspicious keyword detection, and category identification.

The project was developed as part of an AI/ML training and research initiative and evaluated using multiple benchmark fake news datasets.

---

## Key Features

### Fake News Detection

Classifies news articles and statements into:

* Real News
* Fake News

### Confidence Score Estimation

Displays prediction confidence using class probabilities.

### News Category Detection

Automatically identifies categories such as:

* Politics
* Health
* Technology
* Business
* Finance
* Sports
* Science

### Explainable AI Features

Highlights suspicious words that may contribute to misinformation detection.

### Analytics Dashboard

Provides:

* Total Predictions
* Fake News Count
* Real News Count
* Average Confidence Score

### Multi-Dataset Evaluation

Evaluated using:

* WELFake Dataset
* LIAR Dataset
* Kaggle Fake and Real News Dataset

---

## Datasets Used

### 1. WELFake Dataset

Source:
https://huggingface.co/datasets/davanstrien/WELFake

Statistics:

* Total Samples: 72,134
* Fake News: 37,106
* Real News: 35,028

Label Mapping:

* 0 = Real News
* 1 = Fake News

---

### 2. LIAR Dataset

Source:
https://www.cs.ucsb.edu/~william/data/liar_dataset.zip

Statistics:

* Total Samples: 12,791
* Political Statements
* Six Original Credibility Labels

Binary Mapping Used:

* Real = true, mostly-true, half-true
* Fake = barely-true, false, pants-fire

---

### 3. Kaggle Fake and Real News Dataset

Statistics:

* Total Samples: 44,898
* Fake News: 23,481
* Real News: 21,417

Files:

* Fake.csv
* True.csv

---

## Machine Learning Pipeline

News Article

↓

Text Cleaning

↓

Stopword Removal

↓

TF-IDF Vectorization

↓

Logistic Regression Classifier

↓

Prediction

↓

Confidence Analysis

↓

Category Detection

↓

Explainability Output

---

## Model Performance

### Experimental Results

| Dataset          | Accuracy | Precision | Recall | F1 Score |
| ---------------- | -------- | --------- | ------ | -------- |
| WELFake          | 95.07%   | 94.54%    | 95.86% | 95.20%   |
| LIAR             | 61.39%   | 55.89%    | 60.34% | 58.03%   |
| Kaggle Fake Real | 98.98%   | 99.40%    | 98.64% | 99.02%   |

### Observations

* Highest accuracy achieved on Kaggle Fake and Real News Dataset.
* Strong performance observed on WELFake dataset.
* Lower performance on LIAR dataset due to short political statements and limited contextual information.
* Demonstrates robustness across multiple fake news benchmarks.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* NLTK
* Joblib
* Streamlit
* Matplotlib
* Hugging Face Datasets

---

## Project Structure

```text
fake-news-detection-nlp/

├── data/
│   ├── LIAR_dataset/
│   ├── Fake.csv
│   └── True.csv
│
├── models/
│   └── fake_news_model.pkl
│
├── results/
│   ├── dataset_comparison_results.csv
│   ├── confusion_matrix_welfake.png
│   ├── confusion_matrix_liar.png
│   └── confusion_matrix_kaggle_fake_real.png
│
├── train_model.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/skdubey1983/fake-news-detection-nlp.git

cd fake-news-detection-nlp
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Train Model

```bash
python train_model.py
```

Outputs generated:

* Trained Model
* Dataset Comparison Report
* Confusion Matrices

---

## Run Application

```bash
streamlit run app.py
```

---

## Example Prediction

### Input

Scientists discover a magical leaf that instantly cures cancer, diabetes, and heart disease in one day.

### Output

Prediction: FAKE NEWS

Fake Probability: 86.25%

Real Probability: 13.75%

Confidence Level: High Confidence

Detected Category: Health

Suspicious Words:

* disease
* heart
* day

---

## Future Enhancements

* BERT-based Fake News Detection
* RoBERTa and Transformer Models
* Large Language Model Integration
* Real-Time News Verification APIs
* Multilingual Fake News Detection
* Multimodal Analysis (Text + Image + Video)
* SHAP/LIME Explainable AI
* Cloud Deployment

---

## GitHub Repository

Repository:
https://github.com/skdubey1983/fake-news-detection-nlp

---

## Authors

**Shiv Kishan Dubey and Sri Nath Dwivedi**

AI/ML Project – Fake News Detection using NLP and Machine Learning

Developed during Industry Training under ASDL Group.

## License

This project is intended for educational, academic, and research purposes.
