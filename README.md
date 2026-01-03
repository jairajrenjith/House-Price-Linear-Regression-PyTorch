# House Price Linear Regression using PyTorch

This project implements a multivariate Linear Regression model using PyTorch tensors to predict house prices.
The work follows a deep learning approach and was completed entirely in Google Colab as part of a CognitiveClass.ai assignment.

---

## Problem Statement
Build and train a Linear Regression model using a CSV dataset and PyTorch tensors to understand the fundamentals of deep learning–based regression.

---

## Dataset
The dataset contains multiple numerical features related to housing properties:
- Distance to city center
- Location score
- Number of bedrooms
- Square footage
- Year built

The target variable is the house price.

---

## Approach
- Loaded the CSV dataset using pandas
- Converted input features and target values into PyTorch tensors
- Built a Linear Regression model using `torch.nn.Module`
- Trained the model using the Stochastic Gradient Descent (SGD) optimizer
- Evaluated the model using a training loss curve and prediction comparison

---

## Tools and Technologies
- Python
- PyTorch
- Google Colab
- Pandas
- Matplotlib

---

## File Structure

```
House-Price-Linear-Regression-PyTorch/
├── linear_regression_pytorch.ipynb
├── dataset.csv
├── cognitiveclass_progress.png
└── README.md
```

---

## Results
- Training loss decreases steadily across epochs
- Predicted prices closely follow the actual house prices

---

## Proof of Completion
A screenshot showing 100% completion of the Linear Regression course on CognitiveClass.ai is included in this repository.

---

## Conclusion
This project demonstrates how a deep learning–style Linear Regression model can be implemented using PyTorch tensors and trained effectively on real-world numerical data.

---

By Jairaj R.
