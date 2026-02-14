# Implementation of Multivariate Linear Regression
## Aim
To write a python program to implement multivariate linear regression and predict the output.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation / Moodle-Code Runner
## Algorithm:
### Step1
<br> Import the necessary Python libraries for data processing and machine learning.

### Step2
<br>Read the data from a CSV file into a DataFrame for processing.

### Step3
<br>Separate the input (independent variables) and output (dependent variable) from the dataset.

### Step4
<br>Initialize a linear regression model from the machine learning library.

### Step5
<br>Fit the model using the input features and target to learn their relationship.

## Step6
Retrieve and display the coefficients and intercept that the model has learned.

## Step7
Use the trained model to predict the target value for a new set of input features.

## Step8
Run the program.

## Program:
```
# Program to find the solution of a matrix using Gaussian Elimination.
# Developed by: Aniruth S
# RegisterNumber: 212225040020

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split

# -----------------------------
# Load the Boston dataset properly
# -----------------------------
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)

# Correct reshaping (VERY IMPORTANT)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# Remove any NaN values (safety check)
mask = ~(np.isnan(data).any(axis=1) | np.isnan(target))
X = data[mask]
y = target[mask]

# -----------------------------
# Split into train and test sets
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=1
)

# -----------------------------
# Create and train Linear Regression model
# -----------------------------
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)

# -----------------------------
# Print Results
# -----------------------------
print("Coefficients:", reg.coef_)
print("Variance score:", reg.score(X_test, y_test))

# -----------------------------
# Plot Residual Errors
# -----------------------------
plt.style.use("fivethirtyeight")

# Training residuals
plt.scatter(
    reg.predict(X_train),
    reg.predict(X_train) - y_train,
    color="green",
    s=10,
    label="Train data",
)

# Testing residuals
plt.scatter(
    reg.predict(X_test),
    reg.predict(X_test) - y_test,
    color="blue",
    s=10,
    label="Test data",
)

# Zero residual line
plt.hlines(y=0, xmin=0, xmax=50, linewidth=2)

plt.legend(loc="upper right")
plt.title("Residual errors")
plt.show()
```
## Output:

<img width="891" height="736" alt="{1A081BEF-6AF4-47DC-9B44-2B72EB20E27D}" src="https://github.com/user-attachments/assets/48fb0ec7-6469-482d-a1cd-70d9f53ebf49" />


<br>

## Result
Thus the multivariate linear regression is implemented and predicted the output using python program.
