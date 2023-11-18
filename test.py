"""
Developer: Sanjib Sonowal
Email: sanjib.sonowal@gmail.com
Web: www.sanjibsonowal.com
Note: Please email for any collaboration.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(__file__)


def logistic_regression_predict():
    # Predict using Logistic Regression
    df = pd.read_csv('model.csv')
    X = df.drop(columns=["Credit_Score"])
    y = df["Credit_Score"]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    y_predict = lr.predict(x_test)
    pd.DataFrame({"actual_value": y_test, "predicted_value": y_predict})
    # Print prediction accuracy score
    score = accuracy_score(y_test, y_predict)
    print(score)
    return score


def decision_tree_predict():
    df = pd.read_csv('model.csv')
    X = df.drop(columns=["Credit_Score"])
    y = df["Credit_Score"]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    dt = DecisionTreeClassifier()
    dt.fit(x_train, y_train)
    y_predict = dt.predict(x_test)
    pd.DataFrame({"actual_value": y_test, "predicted_value": y_predict})
    # Print prediction accuracy score
    score = accuracy_score(y_test, y_predict)
    print(score)
    return score


def random_forest_predict():
    df = pd.read_csv('model.csv')
    X = df.drop(columns=["Credit_Score"])
    y = df["Credit_Score"]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)
    y_predict = rf.predict(x_test)
    pd.DataFrame({"actual_value": y_test, "predicted_value": y_predict})
    # Print prediction accuracy score
    score = accuracy_score(y_test, y_predict)
    print(score)
    return score


if __name__ == "__main__":
    logistic_regression_predict()
