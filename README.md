# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Prepare and Scale the Data 
2.Initialize Model Parameters (theta) 
3.Perform Gradient Descent to Optimize Parameters 
4.Make Predictions and Scale Back 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Divya R V
RegisterNumber: 212223100005 
*/
```
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Linear Regression Function
def linear_regression(X1, y, learning_rate=0.01, num_iters=1000):
    # Add a column of ones to X1 for the intercept term
    X = np.c_[np.ones(len(X1)), X1]

    # Initialize theta with zeros
    theta = np.zeros(X.shape[1]).reshape(-1, 1)

    # Perform gradient descent
    for _ in range(num_iters):
        # Calculate predictions
        predictions = X.dot(theta).reshape(-1, 1)

        # Calculate errors
        errors = (predictions - y).reshape(-1, 1)

        # Update theta using gradient descent
        theta -= learning_rate * (1 / len(X1)) * X.T.dot(errors)

    return theta

# Load data
data = pd.read_csv('50_Startups.csv')

# Select feature matrix X and target vector y
X = data.iloc[1:, :-2].values
y = data.iloc[1:, -1].values.reshape(-1, 1)

# Convert data to float type
X1 = X.astype(float)

# Standardize the data
scaler = StandardScaler()
X1_Scaled = scaler.fit_transform(X1)
Y1_Scaled = scaler.fit_transform(y)

# Train the linear regression model
theta = linear_regression(X1_Scaled, Y1_Scaled)

# New data for prediction
new_data = np.array([165349.2, 136897.8, 471784.1]).reshape(-1, 1)

# Standardize the new data point
new_Scaled = scaler.fit_transform(new_data)

# Make prediction
prediction = np.dot(np.append(1, new_Scaled), theta)

# Reshape prediction to column vector
prediction = prediction.reshape(-1, 1)

# Inverse scale the prediction
pre = scaler.inverse_transform(prediction)

# Print the predicted value
print(f"Predicted value: {pre}")

```


## Output:
![linear regression using gradient descent](sam.png)

![Screenshot 2025-03-12 034202](https://github.com/user-attachments/assets/9a127a7f-5212-404d-9467-28f83fbb5fc2)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
