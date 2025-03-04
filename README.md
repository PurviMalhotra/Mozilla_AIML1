# Rental Price Prediction & Sentiment Analysis Models
## 1.Rental Price Prediction
## Intoduction

This project aims to perform data visualization and generate key insights from a given dataset. Additionally, a machine learning model is developed on Random Forest algorithm to predict rental prices based on various features such as location, property size, number of bedrooms, and amenities.

## Key Skills
- Python
- Numpy
- Matplotlib
- Pandas
- Scikit-Learn
- Plotly
- Seaborn
- Streamlit

## Installation

To run this project, you need to install the following packages:

```python
pip install numpy
pip install pandas
pip install scikit-learn==3.5.0
pip install xgboost
pip install plotly
pip install matplotlib
pip install seaborn
pip install streamlit
```
## Project Workflow
1.Data Exploration & EDA:Conduct Exploratory Data Analysis (EDA) to identify trends, correlations, and anomalies.

2.Data Visualization:

- Rental Price Distribution: Understand the overall pricing trend.

- Feature Correlation: Analyze how rental prices correlate with factors like location, property size, and amenities.

- Geographic rent variations

3.Preprocess the dataset:
- Data Preprocessing

- Handle missing data using imputation techniques.

- Normalize numerical features for better model performance.

- Encode categorical features using One-Hot Encoding or Label Encoding.

- Split the dataset into training and testing sets.

4. Model Training & Evaluation
- Using Random Forest model
- Train models and fine-tune hyperparameters.
- Evaluate model performance using metrics such as RMSE, MAE, and R-squared.
- Interpret model results and provide actionable insights based on findings.

5. Insights & Interpretation
- Identify key drivers of rental price variations.
- Provide actionable insights to stakeholders.
- Discuss model insights and recommendations.
  
## Usage

1. Load the dataset.
2. Run the EDA and visualization scripts.
3. Preprocess the dataset.
4.Train and evaluate the machine learning model.
5.Generate insights and recommendations.


## 2.Sentiment Analysis Model
## Introduction

This project implements a Sentiment Analysis Model using Support Vector Machines (SVMs). The model classifies text data into positive, negative, or neutral sentiment categories based on given datasets. The project includes text preprocessing, feature extraction, and model training for sentiment classification.

## Key Skills
- Python
- Numpy
- Matplotlib
- Pandas
- Scikit-Learn
- Plotly
- Seaborn
- Streamlit

## Installation

To run this project, you need to install the following packages:

```python
pip install numpy
pip install pandas
pip install scikit-learn==3.5.0
pip install xgboost
pip install plotly
pip install matplotlib
pip install seaborn
pip install streamlit
```

## Project Workflow
1. Data Collection
- Load sentiment-labeled datasets (e.g., Twitter data, IMDB reviews, etc.).
- Perform exploratory data analysis (EDA) to understand data distribution and sentiment balance.

2. Data Preprocessing
- Tokenization: Split text into individual words.
- Feature Extraction:
  - TF-IDF (Term Frequency-Inverse Document Frequency)

3. Model Training

- Split data into training and testing sets.
- Train an SVM classifier with kernel selection (Linear, RBF, or Polynomial).
- Optimize hyperparameters using Grid Search or Cross-Validation.

4. Model Evaluation
- Assess model performance using:
- Accuracy
- Precision, Recall, and F1-score
- Confusion Matrix

5. Prediction & Insights
Use the trained model to classify new text samples.
Identify key sentiment trends in the dataset.
  
## Usage

1. Load the dataset.
2. Preprocess the text data.
3. Extract features using TF-IDF.
4. Train and optimize the SVM classifier.
5. Evaluate and analyze model predictions.
