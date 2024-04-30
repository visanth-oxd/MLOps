# Machine Learning Pipeline for Iris Classification with Logistic Regression

This project contains a Python script that trains a Logistic Regression model on the Iris dataset. The model is then evaluated and saved for future use.

ç Requirements

- Python 3.6 or later
- scikit-learn
- joblib

You can install the required packages using pip:

```bash
pip install scikit-learn joblib
```

### Usage

The `main.py` script trains a Support Vector Machine (SVM) model on the Wine dataset and evaluates its performance. Here's a step-by-step breakdown of what the script does:

1. **Load the Wine dataset**: The Wine dataset is loaded directly from scikit-learn's datasets module.

2. **Split the dataset**: The dataset is split into a training set and a test set, with 70% of the data used for training and 30% used for testing.

3. **Create an SVM classifier**: An SVM classifier with a linear kernel is created.

4. **Train the SVM**: The SVM is trained on the training data, which consists of the features and labels from the Wine dataset.

5. **Make predictions**: The trained SVM is used to make predictions on the test data.

6. **Evaluate the SVM**: The accuracy of the SVM is calculated by comparing its predictions on the test data to the actual labels of the test data. The accuracy is printed to the console.

To run the script, use the command `python main.py` in your terminal.

