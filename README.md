# Enhancing Phishing Email Detection with Sentiment Analysis: A Hybrid Approach
This is my Thesis

# Phishing Email Detection Using DistilBERT and SVM

This project implements a phishing email detection system using a hybrid approach of **DistilBERT** for sentiment analysis and **Support Vector Machine (SVM)** for classification. The model analyzes email content for phishing indicators and uses sentiment-aware embeddings to improve classification accuracy.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)

## Overview

Phishing attacks pose significant cybersecurity threats, targeting individuals and organizations to steal sensitive information. This project leverages machine learning and natural language processing (NLP) to detect phishing emails. By combining sentiment analysis (using DistilBERT embeddings) with SVM classification, the model can effectively distinguish between legitimate and phishing emails.

## Features

- **Sentiment-Aware Embeddings**: DistilBERT is used to capture the sentiment and context within emails, providing rich feature representations.
- **SVM Classification**: A Support Vector Machine (SVM) classifier uses the embeddings to categorize emails as phishing or legitimate.
- **Error Analysis and Misclassification Review**: Includes confusion matrix and classification report to analyze false positives and false negatives.
- **Cross-Validation Support**: Allows for reliable evaluation of model performance.

## Dataset

The dataset for this project is available on [Kaggle](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset/data?select=phishing_email.csv). It includes a collection of phishing and legitimate emails, with key text features and labels to identify each email type. Make sure to download and place it in the appropriate directory as specified in the installation steps.

## Architecture

The project consists of the following key stages:

1. **Data Preprocessing**: Text cleaning, tokenization, stopword removal, and other preprocessing tasks.
2. **Feature Extraction**: DistilBERT is used to generate embeddings for each email text.
3. **Classification**: SVM uses these embeddings to classify emails as phishing or legitimate.
4. **Evaluation**: Metrics like accuracy, precision, recall, and F1-score, as well as a confusion matrix, are used to evaluate performance.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/phishing-email-detection.git
   cd phishing-email-detection
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.7 or above. Use the following command to install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Dataset**:
   Place the downloaded dataset (`phishing_email.csv`) in the `data/` directory or update the path in the code as needed.

4. **Download DistilBERT Model**:
   The code will automatically download the required `DistilBERT` model. Alternatively, you can manually download the model from [Hugging Face](https://huggingface.co/distilbert-base-uncased) and adjust the path if needed.

## Usage

1. **Run Preprocessing**:
   Ensure the dataset is preprocessed to clean the email text and remove unwanted symbols or stopwords. This step is automated within the main script.

2. **Train and Evaluate Model**:
   Run the main training script to perform feature extraction, classification, and evaluation.
   ```bash
   python main.py
   ```

3. **View Results**:
   - After training, the classification report and confusion matrix will display in the console.
   - The trained model will be saved to the specified directory for later use.

## Results

The model has demonstrated high performance with accuracy, precision, and recall metrics around 97%. False positives and false negatives were analyzed to fine-tune the model and improve its robustness. For further details, please refer to the `results/` directory, which includes sample output metrics and confusion matrix visualizations.

| Metric        | Value |
|---------------|-------|
| Accuracy      | 97%   |
| Precision     | 97%   |
| Recall        | 97%   |
| F1-Score      | 97%   |
