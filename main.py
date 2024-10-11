import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

churn_df = pd.read_csv("ChurnData.csv")
churn_df.head()
churn_df = churn_df[
    [
        "tenure",
        "age",
        "address",
        "income",
        "ed",
        "employ",
        "equip",
        "callcard",
        "wireless",
        "churn",
    ]
]
churn_df["churn"] = churn_df["churn"].astype("int")
X = np.asarray(
    churn_df[["tenure", "age", "address", "income", "ed", "employ", "equip"]]
)  # input features, independent variables
y = np.asarray(churn_df["churn"])  # dependent variable vector
X = preprocessing.StandardScaler().fit(X).transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
LR = LogisticRegression(C=0.01, solver="liblinear").fit(X_train, y_train)
# print(LR)
yhat = LR.predict(X_test)
# print(yhat)
yhat_prob = LR.predict_proba(X_test)
print(yhat_prob)
