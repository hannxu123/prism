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

    info_df.loc[info_df['loan amount'] == 'unknown', 'loan amount'] = '20000'
    info_df.loc[info_df['term'] == 'unknown', 'term'] = '36'
    info_df.loc[info_df['employ years'] == 'unknown', 'employ years'] = '3'
    info_df.loc[info_df['home ownership'] == 'unknown', 'home ownership'] = 'OWN'
    info_df.loc[info_df['annual income'] == 'unknown', 'annual income'] = '60000'
    info_df.loc[info_df['interest rate'] == 'unknown', 'interest rate'] = '14.0'
    info_df.loc[info_df['purpose'] == 'unknown', 'purpose'] = 'car'
    info_df.loc[info_df['debt to income ratio'] == 'unknown', 'debt to income ratio'] = '0.35'
    info_df.loc[info_df['revolving util ratio'] == 'unknown', 'revolving util ratio'] = '0.30'
    info_df.loc[info_df['fico score'] == 'unknown', 'fico score'] = '680-710'
    info_df.loc[info_df['inquiry'] == 'unknown', 'inquiry'] = '<=2'
    info_df.loc[info_df['delinquency'] == 'unknown', 'delinquency'] = '0'

    info_df['ID'] = np.arange(len(info_df)) + 1
    info_df = info_df[['ID'] + [c for c in info_df.columns if c != 'ID']]

    return info_df

def compute_ece(probs, labels, n_bins=5, pop_pos=0.0487, sample_pos=0.5):
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if sample_pos is None:
        sample_pos = labels.mean()
    w_pos = pop_pos / sample_pos
    w_neg = (1 - pop_pos) / (1 - sample_pos)
    weights = np.where(labels == 1, w_pos, w_neg)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(probs, bin_edges, right=False) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)
    total_w = weights.sum()
    ece = 0.0
    for b in range(n_bins):
        m = (bin_ids == b)
        if not np.any(m):
            continue
        w = weights[m]
        p = probs[m]
        y = labels[m]
        Wb = w.sum()
        conf_bin = (w * p).sum() / Wb
        acc_bin  = (w * y).sum() / Wb
        ece += (Wb / total_w) * abs(acc_bin - conf_bin)
    return ece
