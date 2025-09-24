import openai 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import roc_curve
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from torch.utils.data import Dataset, DataLoader


def sklearn_logistic(X_train, y_train, X_test, y_test):
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(class_weight={0:1, 1:pos_weight}, max_iter=100000)
    model.fit(X_train_scaled, y_train)
    X_test_scaled = scaler.transform(X_test)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    conf_matrix = confusion_matrix(y_test, y_pred_proba > 0.5)

    print('This is the result of sklearn logistic regression')
    print(f"ROC AUC: {auc:.4f}")
    print(conf_matrix)

    return auc

def random_subset_at_least_k(lst, tg_feature):
    lst2 = np.random.permutation(np.array(lst))
    idx = np.where(lst2 == tg_feature)[0][0]
    bg_list = lst2[0: idx]
    return bg_list.tolist()


import re
def first_number_in_string(s):
    match = re.search(r'\d+(?:\.\d+)?', s)
    if match:
        return float(match.group())
    return None

def prob_to_logit(p):
    return np.log(p / (1 - p))

def logit_to_prob(x):
    return 1 / (1 + np.exp(-x))


def process_df(info_df):


    info_df.loc[info_df['age'] == 'unknown', 'age'] =  '40' 

    info_df.loc[info_df['workclass'] == 'unknown', 'workclass'] = 'Private'
    info_df.loc[info_df['workclass'] == '?', 'workclass'] = 'Not provided'

    info_df.loc[info_df['education level'] == 'unknown', 'education level'] = 'Some-college'
    # info_df.loc[info_df['education years'] == 'unknown', 'education years'] = '10'

    info_df.loc[info_df['marital status'] == 'unknown', 'marital status'] = 'Married-civ-spouse'

    info_df.loc[info_df['occupation'] == 'unknown', 'occupation'] = 'Sales'
    info_df.loc[info_df['occupation'] == '?', 'occupation'] = 'Not provided'

    # info_df.loc[info_df['relationship'] == 'unknown', 'relationship'] = 'Not provided'

    info_df.loc[info_df['capital gain'] == 'unknown', 'capital gain'] = '0'
    info_df.loc[info_df['capital loss'] == 'unknown', 'capital loss'] = '0'
    info_df.loc[info_df['working hours per week'] == 'unknown', 'working hours per week'] = '40'

    info_df.loc[info_df['native country'] == 'unknown', 'native country'] = 'United-States'

    info_df['ID'] = np.arange(len(info_df)) + 1
    info_df = info_df[['ID'] + [c for c in info_df.columns if c != 'ID']]

    return info_df