# House Price Linear Regression using PyTorch

This project implements a multivariate Linear Regression model using PyTorch tensors to predict house prices.
The work follows a deep learning approach and was completed entirely in Google Colab as part of a CognitiveClass.ai assignment.

---

## Problem Statement
Build and train a Linear Regression model using a CSV dataset and PyTorch tensors to understand the fundamentals of deep learning–based regression.

---

## Approach
- Loaded the CSV dataset using pandas
- Converted input features and target values into PyTorch tensors
- Built a Linear Regression model using `torch.nn.Module`
- Trained the model using the Stochastic Gradient Descent (SGD) optimizer
- Evaluated the model using a training loss curve and prediction comparison

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

## Tools and Technologies
- Python
- PyTorch
- Google Colab
- Pandas
- Matplotlib

---

## Model Details
- Model type: Multivariate Linear Regression
- Input layer: Multiple numerical features
- Output layer: Single continuous value (house price)
- Loss function: Mean Squared Error (MSE)
- Optimizer: Stochastic Gradient Descent (SGD)

---

## Training Details
- The model was trained for multiple epochs to ensure convergence
- Loss values were recorded at each epoch
- A loss curve was plotted to visualize learning behavior
- A comparison plot between actual and predicted prices was used for evaluation

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

## Proof of Completion
A screenshot showing 100% completion of the Linear Regression course on CognitiveClass.ai is included in this repository.

---

## How to Run the Notebook
1. Open Google Colab
2. Upload the `linear_regression_pytorch.ipynb` notebook
3. Upload the `dataset.csv` file when prompted
4. Run all cells sequentially to train and evaluate the model

---

## Learning Outcomes
- Gained hands-on experience with PyTorch tensors
- Understood how linear regression fits into deep learning workflows
- Learned how to structure and train models using `torch.nn.Module`
- Practiced working with CSV datasets in machine learning tasks
- Understood the importance of evaluation and visualization in regression problems

---


## Results
- Training loss decreases steadily across epochs
- Predicted prices closely follow the actual house prices

---

## Notes
- No external virtual environment is required
- All experiments were conducted in Google Colab
- The project focuses on clarity and correctness rather than hyperparameter tuning

---

## Conclusion
This project demonstrates how a deep learning–style Linear Regression model can be implemented using PyTorch tensors and trained effectively on real-world numerical data.

---

By Jairaj R.
