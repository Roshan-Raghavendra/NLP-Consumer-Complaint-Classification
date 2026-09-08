# NLP Consumer Complaint Classification

An end-to-end Machine Learning and Natural Language Processing project that automatically classifies consumer complaints into financial product categories using TF-IDF and Machine Learning.

The project also includes sentiment analysis, complaint priority detection, and a Streamlit web application for real-time predictions.

---

## 📌 Project Overview

Financial institutions receive a large number of customer complaints every day. Manually categorizing these complaints can be time-consuming and inconsistent.

This project uses Natural Language Processing (NLP) and Machine Learning to automatically analyze complaint text and predict the appropriate financial product category.

###  The system can:

- Classify consumer complaints into financial product categories
- Convert textual complaints into numerical features using TF-IDF
- Compare multiple Machine Learning algorithms
- Perform hyperparameter tuning using GridSearchCV
- Analyze complaint sentiment
- Assign complaint priority
- Provide real-time predictions through a Streamlit application

---

## 🎯 Problem Statement

Build an NLP-based classification system that can automatically categorize consumer complaints based on their textual descriptions.

The system aims to reduce manual classification effort and provide a scalable approach for processing large volumes of customer complaints.

---

## 🧠 Machine Learning Pipeline

```text
Consumer Complaint
        ↓
Data Cleaning
        ↓
Missing Value & Duplicate Removal
        ↓
Text Preprocessing
        ↓
Stopword Removal
        ↓
Lemmatization
        ↓
TF-IDF Feature Extraction
        ↓
Train/Test Split
        ↓
Multiple ML Models
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

Complaint Categories

The project contains five major complaint categories:

Credit Reporting
Debt Collection
Mortgages and Loans
Credit Card
Retail Banking

Note: The original dataset is not included in this GitHub repository because of its size. The application and trained models are included.


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

TF-IDF helps represent important words and phrases numerically so that Machine Learning algorithms can learn patterns from complaint text.


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

Linear SVM achieved the best baseline performance.

GridSearchCV with 3-fold cross-validation was used to tune the C parameter.

Tested Values
C = 0.5
C = 1
C = 1.5
C = 2

Best Parameter
C = 0.5

Final Model Performance
Metric	Score
Accuracy	85.76%
Precision	85.97%
Recall	85.76%
Weighted F1 Score	85.82%

The tuned Linear SVM was selected as the final complaint classification model.

💬 Sentiment Analysis

VADER sentiment analysis was integrated as an additional NLP component.

The system categorizes complaint text as:

Positive
Neutral
Negative

It also generates a compound sentiment score.

Sentiment analysis is an additional analytical feature and is separate from the primary complaint-category classification model.

🚨 Complaint Priority Detection

A rule-based priority system was implemented using complaint keywords and sentiment.

The system assigns four priority levels:

Critical
High
Medium
Low
Critical Indicators

Examples include:

fraud
stolen
unauthorized
identity theft
scam
emergency

This component can help identify complaints that may require faster attention.


🌐 Streamlit Application

A Streamlit web application was developed for real-time complaint analysis.

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
Multinomial Naive Bayes
Random Forest
Natural Language Processing
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
├── .gitignore
├── README.md
└── requirements.txt


🚀 Installation
1. Clone the Repository
git clone https://github.com/YOUR-USERNAME/NLP-Consumer-Complaint-Classification.git

2. Navigate to the Project
cd NLP-Consumer-Complaint-Classification

3. Create a Virtual Environment
python -m venv venv

4. Activate the Environment
venv\Scripts\activate

5. Install Dependencies
pip install -r requirements.txt


▶️ Run the Application

Start the Streamlit application:
streamlit run app/app.py

The application will open in your browser.

Example Predictions
Example 1

Input:
Someone made an unauthorized transaction from my bank account and I did not make this transaction.

Output:

Complaint Category: retail_banking
Priority: Critical
Sentiment: Neutral
Example 2

Input:

I am having serious problems with my mortgage payment and need help immediately.

Output:

Complaint Category: mortgages_and_loans
Priority: High
Sentiment: Negative

📈 Key Results

The final tuned Linear SVM achieved:

85.76% Accuracy
85.82% Weighted F1 Score
85.97% Precision
85.76% Recall

The model classifies consumer complaints across five financial product categories.

🔮 Future Improvements
Fine-tune transformer-based models such as BERT
Improve domain-specific sentiment analysis
Add prediction confidence scores
Implement a database for complaint storage
Add automated complaint routing
Deploy the application to a cloud platform
Add multilingual complaint support
Explore advanced deep learning architectures for text classification
👨‍💻 Author

Roshan Raghavendra

⭐ Project Highlights
End-to-end NLP pipeline
Text preprocessing and feature engineering
TF-IDF based text representation
Multiple Machine Learning model comparison
Hyperparameter tuning using GridSearchCV
85.76% classification accuracy
85.82% weighted F1 score
Sentiment analysis
Rule-based complaint prioritization
Real-time Streamlit application
Saved trained ML model and TF-IDF vectorizer

