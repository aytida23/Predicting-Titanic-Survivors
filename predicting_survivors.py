"""
This is the code implementation of Classification's Model.
"""

# Logistic Regression

#Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import titanic
from titanic import *

# Predictions
# Training and Testing data

X_train = train_data.drop('Survived', axis = 1)
Y_train = train_data['Survived']
X_test = test_data.drop('PassengerId', axis = 1).copy()

# Running our classifier
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
accuracy = round(decision_tree.score(X_train, Y_train) * 100, 2)
print("Model Accuracy is : ",accuracy)

# Create a csv data file with results
submission = pd.DataFrame({
        "PassengerId": test_data['PassengerId'],
        "Survived": Y_pred
        })

submission.to_csv('survivors_submission.csv', index = False)