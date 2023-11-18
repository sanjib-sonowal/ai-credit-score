"""
Developer: Sanjib Sonowal
Email: sanjib.sonowal@gmail.com
Web: www.sanjibsonowal.com
Note: Please email for any collaboration.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor


BASE_DIR = os.path.dirname(__file__)


def train():
    # import the dataset
    df = pd.read_csv('training_set.csv')
    df = df.drop(columns=["ID", "Customer_ID", "Name", "SSN", "Type_of_Loan", "Credit_History_Age"])

    # data cleaning
    df["Age"] = df["Age"].str.replace("_", "")
    df["Age"] = df["Age"].astype(int)
    df["Occupation"] = df["Occupation"].replace("_______", np.nan)
    df["Annual_Income"] = df["Annual_Income"].str.replace("_", "")
    df["Annual_Income"] = df["Annual_Income"].astype(float)
    df["Num_of_Loan"] = df["Num_of_Loan"].str.replace("_", "")
    df["Num_of_Loan"] = df["Num_of_Loan"].astype(int)
    df["Num_of_Delayed_Payment"] = df["Num_of_Delayed_Payment"].str.replace("_", "")
    df["Num_of_Delayed_Payment"] = df["Num_of_Delayed_Payment"].astype(float)
    df["Credit_Score"] = df["Credit_Score"].replace(["Poor", "Standard", "Good"], [0, 1, 2])
    df["Monthly_Balance"] = df["Monthly_Balance"].str.replace("_", "")
    df["Monthly_Balance"] = df["Monthly_Balance"].astype(float)
    df["Payment_Behaviour"] = df["Payment_Behaviour"].replace("!@9#%8", np.nan)
    df["Amount_invested_monthly"] = df["Amount_invested_monthly"].str.replace("_", "")
    df["Amount_invested_monthly"] = df["Amount_invested_monthly"].astype(float)
    df["Payment_of_Min_Amount"] = df["Payment_of_Min_Amount"].replace("NM", "No")
    df["Payment_of_Min_Amount"] = df["Payment_of_Min_Amount"].replace(["Yes", "No"], [1, 0])
    df["Outstanding_Debt"] = df["Outstanding_Debt"].str.replace("_", "")
    df["Outstanding_Debt"] = df["Outstanding_Debt"].astype(float)
    df["Credit_Mix"] = df["Credit_Mix"].replace("_", np.nan)
    df["Credit_Mix"] = df["Credit_Mix"].replace(["Standard", "Good", "Bad"], [1, 2, 0])
    df["Changed_Credit_Limit"] = df["Changed_Credit_Limit"].replace("_", np.nan)
    df["Changed_Credit_Limit"] = df["Changed_Credit_Limit"].astype(float)

    # Fill null values
    df = df.fillna(method="ffill")
    df = df.fillna(method="bfill")

    # Perform One Hot Encoding for categorical features of the dataframe
    le = LabelEncoder()
    df["Month"] = le.fit_transform(df["Month"])
    df["Occupation"] = le.fit_transform(df["Occupation"])
    df["Payment_Behaviour"] = le.fit_transform(df["Payment_Behaviour"])

    # Feature Selection.
    # Selecting the features using VIF. VIF should be less than 5.
    # Here all the features is having VIF value less than 5, so we will take all the features.
    col_list = []

    for col in df.columns:
        if (df[col].dtype != "object") & (col != "Credit_Score"):
            col_list.append(col)

    X = df[col_list]
    vif_data = pd.DataFrame()
    vif_data["features"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    print(vif_data)

    df.to_csv(BASE_DIR + "/model.csv", index=False)

    return True
