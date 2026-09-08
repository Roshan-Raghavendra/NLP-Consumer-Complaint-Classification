# NLP Consumer Complaint Classification

A Machine Learning and Natural Language Processing project that automatically classifies consumer complaints into financial product categories using TF-IDF and multiple machine learning algorithms.

The project also includes sentiment analysis, complaint priority detection, and a Streamlit web application for real-time predictions.

---

## 📌 Project Overview

Financial institutions receive a large number of customer complaints every day. Manually categorizing these complaints can be time-consuming and inconsistent.

This project uses Natural Language Processing (NLP) and Machine Learning to automatically analyze complaint text and predict the appropriate financial product category.

The system can:

- Classify consumer complaints into financial product categories
- Convert textual complaints into numerical features using TF-IDF
- Compare multiple Machine Learning algorithms
- Perform hyperparameter tuning
- Analyze complaint sentiment
- Assign complaint priority
- Provide real-time predictions through a Streamlit application

---

## 🎯 Problem Statement

Build an NLP-based classification system that can automatically categorize consumer complaints based on their textual descriptions.

The system should help reduce manual classification effort and provide a scalable approach for processing large volumes of customer complaints.

---

## 🧠 Machine Learning Approach

The project follows the following pipeline:

```text
Consumer Complaint
        ↓
Data Cleaning
        ↓
Missing Value & Duplicate Removal
        ↓
Text Preprocessing
        ↓
Tokenization
        ↓
Stopword Removal
        ↓
Lemmatization
        ↓
TF-IDF Feature Extraction
        ↓
Train/Test Split
        ↓
Machine Learning Models
        ↓
Model Evaluation
        ↓
Hyperparameter Tuning
        ↓
Best Model Selection
        ↓
Real-Time Prediction


📊 Dataset

The dataset contains consumer complaint narratives along with their corresponding financial product categories.

Dataset Features
Feature	Description
product	Target complaint category
narrative	Textual description of the complaint

The dataset contains 162,421 records before cleaning.

The project uses five major complaint categories:

Credit Reporting
Debt Collection
Mortgages and Loans
Credit Card
Retail Banking
🧹 Data Preprocessing

The following preprocessing steps were performed:

Removed unnecessary index column
Removed rows with missing complaint narratives
Removed duplicate records
Converted text to lowercase
Removed special characters and numbers
Removed stopwords
Applied lemmatization
Reset the dataset index

After preprocessing, the cleaned text was used for machine learning.

🔤 TF-IDF Feature Extraction

Term Frequency-Inverse Document Frequency (TF-IDF) was used to convert complaint text into numerical feature vectors.

The implementation uses:

Unigrams and bigrams
Maximum 30,000 features
Minimum document frequency of 2
Maximum document frequency of 95%
Sublinear TF scaling

This representation allows machine learning algorithms to identify important words and phrases within complaint narratives.

🤖 Machine Learning Models

Four classification algorithms were evaluated:

Multinomial Naive Bayes
Logistic Regression
Linear Support Vector Machine
Random Forest
Model Comparison
Model	Accuracy	Precision	Recall	F1 Score
Linear SVM	85.45%	85.62%	85.45%	85.51%
Random Forest	85.19%	85.31%	85.19%	85.23%
Logistic Regression	84.77%	85.50%	84.77%	84.90%
Naive Bayes	82.47%	82.83%	82.47%	82.46%
⚙️ Hyperparameter Tuning

The Linear SVM model achieved the best baseline performance.

GridSearchCV with 3-fold cross-validation was used to tune the C parameter.

Tested values:

C = 0.5
C = 1
C = 1.5
C = 2
Best Parameters
C = 0.5
Final Model Performance
Metric	Score
Accuracy	85.76%
Precision	85.97%
Recall	85.76%
Weighted F1 Score	85.82%

The tuned Linear SVM was selected as the final classification model.

💬 Sentiment Analysis

VADER sentiment analysis was integrated as an additional NLP component.

The system categorizes complaint text as:

Positive
Neutral
Negative

It also generates a compound sentiment score.

Sentiment analysis is used as an additional analytical feature and is separate from the primary complaint-category classification model.

🚨 Complaint Priority

A rule-based priority system was implemented using complaint keywords and sentiment.

The system assigns:

Critical
High
Medium
Low

Examples of critical indicators include:

fraud
stolen
unauthorized
identity theft
scam
emergency

This component can help identify complaints that may require faster attention.

🌐 Streamlit Application

A Streamlit web application was developed to provide real-time complaint analysis.

Users can enter a complaint and receive:

Predicted complaint category
Sentiment
Sentiment score
Complaint priority
Application Screenshots
Example 1

Example 2

Example 3

🛠️ Technologies Used
Programming Language
Python
Machine Learning
Scikit-learn
Linear SVM
Logistic Regression
Naive Bayes
Random Forest
NLP
NLTK
TF-IDF
Stopword Removal
Lemmatization
VADER Sentiment Analysis
Data Processing
Pandas
NumPy
Visualization
Matplotlib
Seaborn
Deployment
Streamlit
Model Persistence
Joblib
📁 Project Structure
NLP-Consumer-Complaint-Classification/
│
├── app/
│   └── app.py
│
├── data/
│   └── complaints_processed.csv
│
├── images/
│   ├── prediction_1.png
│   ├── prediction_2.png
│   └── prediction_3.png
│
├── models/
│   ├── complaint_classifier.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebooks/
│   └── NLP_Consumer_Complaint_Classification.ipynb
│
├── src/
│
├── .gitignore
├── README.md
└── requirements.txt
🚀 Installation

Clone the repository:

git clone <YOUR_GITHUB_REPOSITORY_URL>

Navigate to the project directory:

cd NLP-Consumer-Complaint-Classification

Create a virtual environment:

python -m venv venv

Activate the environment on Windows:

venv\Scripts\activate

Install the required dependencies:

pip install -r requirements.txt
▶️ Run the Application

Start the Streamlit application:

streamlit run app/app.py

The application will open in the browser.

🔍 Example
Input
Someone made an unauthorized transaction from my bank account and I did not make this transaction.
Output
Complaint Category: retail_banking
Priority: Critical
Sentiment: Neutral

Another example:

I am having serious problems with my mortgage payment and need help immediately.

Output:

Complaint Category: mortgages_and_loans
Priority: High
Sentiment: Negative
📈 Key Results

The final tuned Linear SVM achieved:

85.76% Accuracy

85.82% Weighted F1 Score

The model successfully classifies consumer complaints across five financial product categories.

🔮 Future Improvements

Possible future improvements include:

Fine-tuning transformer-based models such as BERT
Improving domain-specific sentiment analysis
Adding confidence scores to predictions
Implementing a database for complaint storage
Adding an automated complaint-routing system
Deploying the application to a cloud platform
Adding multilingual complaint support
Using advanced deep learning architectures for text classification
👨‍💻 Author

Roshan Raghavendra

⭐ Project Highlights
End-to-end NLP pipeline
Text preprocessing and feature engineering
TF-IDF based text representation
Multiple ML model comparison
Hyperparameter tuning using GridSearchCV
85.76% classification accuracy
Sentiment analysis
Rule-based complaint prioritization
Real-time Streamlit application
Saved production-ready ML model