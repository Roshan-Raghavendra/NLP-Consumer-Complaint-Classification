# NLP Consumer Complaint Classification

A Machine Learning and NLP project that classifies consumer complaints into financial product categories based on their text.

The project also includes sentiment analysis, complaint priority detection, and a Streamlit application for real-time predictions.

## Project Overview

The main steps in the project are:

- Data cleaning and text preprocessing
- TF-IDF feature extraction
- Machine Learning model comparison
- Hyperparameter tuning
- Sentiment analysis
- Complaint priority detection
- Streamlit application

## Dataset

The dataset contains **162,421 records** before preprocessing.

The five complaint categories are:

- Credit Reporting
- Debt Collection
- Mortgages and Loans
- Credit Card
- Retail Banking

The original dataset is not included in the repository because of its size.

## Text Preprocessing

The following steps were performed:

- Removed missing and duplicate records
- Removed unnecessary columns
- Converted text to lowercase
- Removed special characters and numbers
- Removed stopwords
- Applied lemmatization

## TF-IDF

TF-IDF was used to convert complaint text into numerical features.

- Unigrams and bigrams
- Maximum 30,000 features
- Minimum document frequency: 2
- Maximum document frequency: 95%
- Sublinear TF scaling

## Machine Learning Models

Four models were compared:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Linear SVM | 85.45% | 85.62% | 85.45% | 85.51% |
| Random Forest | 85.19% | 85.31% | 85.19% | 85.23% |
| Logistic Regression | 84.77% | 85.50% | 84.77% | 84.90% |
| Naive Bayes | 82.47% | 82.83% | 82.47% | 82.46% |

Linear SVM gave the best baseline result.

GridSearchCV was used to tune the SVM.

**Best C value: 0.5**

Final model:

- Accuracy: **85.76%**
- Precision: **85.97%**
- Recall: **85.76%**
- Weighted F1 Score: **85.82%**

The tuned Linear SVM was selected as the final model.

## Sentiment Analysis

VADER was used to identify the sentiment of a complaint.

The application displays:

- Positive
- Neutral
- Negative
- Sentiment score

## Complaint Priority

A rule-based system assigns four priority levels:

- Critical
- High
- Medium
- Low

Keywords such as `fraud`, `stolen`, `unauthorized`, `identity theft`, and `scam` are used as indicators for higher priority.

## Streamlit Application

The Streamlit application allows users to enter a complaint and get:

- Predicted category
- Sentiment
- Sentiment score
- Complaint priority

### Example

**Input:**

```text
Someone made an unauthorized transaction from my bank account and I did not make this transaction.
```

**Output:**

```text
Category: retail_banking
Priority: Critical
Sentiment: Neutral
```

## Project Structure

```text
NLP-Consumer-Complaint-Classification/
│
├── app/
│   └── app.py
├── images/
│   ├── prediction_1.png
│   ├── prediction_2.png
│   └── prediction_3.png
├── models/
│   ├── complaint_classifier.pkl
│   └── tfidf_vectorizer.pkl
├── notebooks/
│   └── NLP_Consumer_Complaint_Classification.ipynb
├── .gitignore
├── README.md
└── requirements.txt
```

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- TF-IDF
- VADER
- Matplotlib
- Seaborn
- Streamlit
- Joblib

## How to Run

Clone the repository:

```bash
git clone https://github.com/Roshan-Raghavendra/NLP-Consumer-Complaint-Classification.git
```

Open the project folder:

```bash
cd NLP-Consumer-Complaint-Classification
```

Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app/app.py
```

## Author

**Roshan Raghavendra**