# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 11:55:46 2024

@author: admin
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
df=pd.read_csv("C:/Users/admin/Downloads/gym_members_exercise_tracking.csv")
df.info()
describe=df.describe()
print(describe)
df.drop(columns=['Workout_Type'], inplace=True)
X=df.iloc[:,:-1]
y=df['BMI']

from sklearn.preprocessing import LabelEncoder
la=LabelEncoder()
X['Gender']=la.fit_transform(X['Gender'])
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)

from sklearn.model_selection import train_test_split as tts
X_train,X_test,y_train,y_test=tts(X,y,test_size=0.2,random_state=0)



scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Step 7: Train a regression model (Linear Regression in this case)
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Step 8: Make predictions on the test set
y_pred = model.predict(X_test_scaled)

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel('Actual BMI')
plt.ylabel('Predicted BMI')
plt.title('Actual vs Predicted BMI')
plt.show()




# Step 9: Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R-squared (R²):", r2)


