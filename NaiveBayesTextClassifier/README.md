# Naïve Bayes Classifier for Presidential Speeches
This project implements a Naïve Bayes classifier for classifying presidential speeches from John F. Kennedy and Lyndon B. Johnson. It also compares the results with the scikit-learn implementation of Naïve Bayes.

## Project Overview
The classifier takes in a dataset of speeches, tokenizes the text, and applies Lidstone smoothing to estimate word likelihoods. The model predicts the author of an unlabeled speech and evaluates its performance using accuracy, F1-score, and a confusion matrix.

## Dataset Structure

The dataset consists of speech text files organized into directories:

data/  
│── kennedy/       # Speeches by John F. Kennedy  
│── johnson/       # Speeches by Lyndon B. Johnson  
│── unlabeled/     # Unlabeled test speeches  

## Dependencies

Ensure you have the following Python libraries installed:

''' pip install numpy pandas matplotlib seaborn scikit-learn '''

## How to Run

Run the script with the following command:

''' python scripts/nb.py -f data/ '''

## Features

- Custom Naïve Bayes Implementation
- Tokenizes and counts word frequencies
- Uses Lidstone smoothing for likelihood estimation
- Predicts authors based on learned priors and likelihoods
- Comparison with scikit-learn
- Uses CountVectorizer for text processing
- Trains and evaluates a Multinomial Naïve Bayes model
- Performance Metrics
- Computes accuracy and F1-score
- Generates a confusion matrix using Seaborn

## Results & Output

### Priors and Likelihoods
The program prints computed prior probabilities and the shape of the likelihood matrix.
### Predictions
The classifier outputs predicted authors for the test data.
### Performance Metrics
Outputs Accuracy, F1, and Confusion Matrix.

## Functions Overview
| Function |	Description |
|:----------|:----------|:----------|
| build_dataframe(folder) |	Loads speech text data into Pandas DataFrames |
| train_nb(df, alpha=0.1) |	Trains the custom Naïve Bayes classifier |
| test(df, vocabulary, priors, likelihoods) |	Makes predictions using the trained model |
| sklearn_nb(training_df, test_df) |	Runs scikit-learn’s Naïve Bayes for comparison |
| get_metrics(true, preds) |	Computes accuracy, F1-score, and confusion matrix |
| plot_confusion_matrix(conf_matrix, labels) |	Visualizes the confusion matrix |
