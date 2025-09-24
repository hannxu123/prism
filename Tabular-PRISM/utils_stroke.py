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
    info_df.loc[info_df['gender'] == '0', 'gender'] = 'Female'
    info_df.loc[info_df['gender'] == '1', 'gender'] = 'Male'
    info_df.loc[info_df['gender'] == 'unknown', 'gender'] = 'Male'

    info_df.loc[info_df['age'] == 'unknown', 'age'] =  '40.0' 

    info_df.loc[info_df['hypertension'] == '0', 'hypertension'] = 'No'
    info_df.loc[info_df['hypertension'] == '1', 'hypertension'] = 'Yes'
    info_df.loc[info_df['hypertension'] == 'unknown', 'hypertension'] = 'No'

    info_df.loc[info_df['heart disease'] == '0', 'heart disease'] = 'No'
    info_df.loc[info_df['heart disease'] == '1', 'heart disease'] = 'Yes'
    info_df.loc[info_df['heart disease'] == 'unknown', 'heart disease'] = 'No'

    info_df.loc[info_df['marital status'] == '0', 'marital status'] = 'Never Married'
    info_df.loc[info_df['marital status'] == '1', 'marital status'] = 'Ever Married'
    info_df.loc[info_df['marital status'] == 'unknown', 'marital status'] = 'Never Married'

    info_df.loc[info_df['residence type'] == '0', 'residence type'] = 'Rural'
    info_df.loc[info_df['residence type'] == '1', 'residence type'] = 'Urban'
    info_df.loc[info_df['residence type'] == 'unknown', 'residence type'] = 'Rural'

    info_df.loc[info_df['average glucose level'] == 'unknown', 'average glucose level'] = '90.0'
    info_df.loc[info_df['Body Mass Index (BMI)'] == 'unknown', 'Body Mass Index (BMI)'] = '24.0' 

    info_df.loc[info_df['work type'] == 'unknown', 'work type'] = 'Private'
    info_df.loc[info_df['smoking status'] == 'unknown', 'smoking status'] = 'never smoked'

    info_df['ID'] = np.arange(len(info_df)) + 1
    info_df = info_df[['ID'] + [c for c in info_df.columns if c != 'ID']]

    return info_df


def process_df_loan(info_df):
    info_df.loc[info_df['HAS_OWN_CAR'] == 0, 'HAS_OWN_CAR'] = 'No car'
    info_df.loc[info_df['HAS_OWN_CAR'] == 1, 'HAS_OWN_CAR'] = 'Owns a car'
    info_df.loc[info_df['HAS_OWN_CAR'] == '0', 'HAS_OWN_CAR'] = 'No car'
    info_df.loc[info_df['HAS_OWN_CAR'] == '1', 'HAS_OWN_CAR'] = 'Owns a car'
    info_df.loc[info_df['HAS_OWN_CAR'] == 'unknown', 'HAS_OWN_CAR'] = 'No car'

    info_df.loc[info_df['HAS_OWN_REALTY'] == 0, 'HAS_OWN_REALTY'] = 'No real estate'
    info_df.loc[info_df['HAS_OWN_REALTY'] == 1, 'HAS_OWN_REALTY'] = 'Owns real estate'
    info_df.loc[info_df['HAS_OWN_REALTY'] == '0', 'HAS_OWN_REALTY'] = 'No real estate'
    info_df.loc[info_df['HAS_OWN_REALTY'] == '1', 'HAS_OWN_REALTY'] = 'Owns real estate'
    info_df.loc[info_df['HAS_OWN_REALTY'] == 'unknown', 'HAS_OWN_REALTY'] = 'No real estate'

    info_df.loc[info_df['INCOME_TYPE'] == 'unknown', 'INCOME_TYPE'] = 'Working'
    info_df.loc[info_df['EDUCATION'] == 'unknown', 'EDUCATION'] = 'Incomplete higher'
    info_df.loc[info_df['CONTRACT_TYPE'] == 'unknown', 'CONTRACT_TYPE'] = 'Cash loans'
    info_df.loc[info_df['FAMILY_STATUS'] == 'unknown', 'FAMILY_STATUS'] = 'Single / not married'
    info_df.loc[info_df['OCCUPATION_TYPE'] == 'unknown', 'OCCUPATION_TYPE'] = 'Laborers'
    num_defaults = {
        'AGE': 40.0,  # mean
        'YEARS_EMPLOYED': 4.8,  # mean
        'NUM_CHILDREN': 0.0,
        'NUM_NON_CHILD_FAM_MEMBERS': 1.0,
        'ANNUAL_INCOME': 160000,
        'ANNUAL_LOAN_PAYMENT': 26000,
        'LOAN_CREDIT': 560000,
        'GOODS_PRICE': 500000,
        'PREVIOUS_LOAN_RATING': 3.0,
        'BUREAU_RATING': 3.0,
        'SOCIAL_SURROUNDING_RISK_RATING': 2.0
    }

    for col, default_val in num_defaults.items():
        info_df.loc[info_df[col] == 'unknown', col] = default_val
        info_df[col] = info_df[col].astype(float)

    info_df['ID'] = np.arange(len(info_df)) + 1
    info_df = info_df[['ID'] + [c for c in info_df.columns if c != 'ID']]

    return info_df


def preprocess_df_adult(info_df):
    # workclass 映射（工作类别）
    workclass_map = {
        'Private': 'Private company',
        'Self-emp-not-inc': 'Self-employed (not incorporated)',
        'Self-emp-inc': 'Self-employed (incorporated)',
        'Local-gov': 'Local government',
        'State-gov': 'State government',
        'Federal-gov': 'Federal government',
        'Unknown': 'Unknown',
    }
    info_df['workclass'] = info_df['workclass'].replace(workclass_map)

    # education 映射（教育程度）
    education_map = {
        'HS-grad': 'High school graduate',
        'Some-college': 'Some college but no degree',
        'Bachelors': 'Bachelor’s degree',
        'Masters': 'Master’s degree',
        'Doctorate': 'Doctorate degree (PhD/EdD/etc.)',
        'Assoc-voc': 'Associate degree (vocational)',
        'Assoc-acdm': 'Associate degree (academic)',
        'Prof-school': 'Professional school degree (JD/MD/LLM/etc.)',
        '11th': '11th grade',
        '10th': '10th grade',
        '9th': '9th grade',
        '12th': '12th grade (completed)',
        '5th-6th': '5th to 6th grade',
    }
    info_df['education'] = info_df['education'].replace(education_map)

    # marital.status 映射（婚姻状况）
    marital_map = {
        'Married-civ-spouse': 'Married (civilian spouse)',
        'Married-spouse-absent': 'Married (spouse absent)',
        'Never-married': 'Never married',
        'Divorced': 'Divorced',
        'Separated': 'Separated',
        'Widowed': 'Widowed',
    }
    info_df['marital.status'] = info_df['marital.status'].replace(marital_map)

    # occupation 映射（职业）
    occupation_map = {
        'Exec-managerial': 'Executive / Managerial',
        'Prof-specialty': 'Professional specialty',
        'Craft-repair': 'Craft / Repair',
        'Sales': 'Sales',
        'Other-service': 'Other service',
        'Adm-clerical': 'Administrative clerical',
        'Machine-op-inspct': 'Machine operator / Inspector',
        'Transport-moving': 'Transport / Moving',
        'Handlers-cleaners': 'Handlers / Cleaners',
        'Tech-support': 'Technical support',
        'Protective-serv': 'Protective service',
        'Farming-fishing': 'Farming / Fishing',
        'Priv-house-serv': 'Private household service',
        'Unknown': 'Unknown',
    }
    info_df['occupation'] = info_df['occupation'].replace(occupation_map)


    int_columns = ["age", "capital.gain", "capital.loss", "hours.per.week"]

    for col in int_columns:
        info_df[col] = info_df[col].astype(float).astype(int)

    return info_df


def process_df_adult(info_df):
    defaults = {
        # 数值型字段
        'age': 40,              # mean
        'capital.gain': 0,      # 大部分人是0
        'capital.loss': 0,      # 大部分人是0
        'hours.per.week': 40,   # mean
        
        # 类别型字段（mode-based）
        'workclass': 'Private company',                 # 最常见
        'education': 'Some college but no degree',                 # 最常见
        'marital.status': 'Never married', # 最常见
        'occupation': 'Professional specialty',         # 最常见
        'relationship': 'Husband',              # 最常见
        'native.country': 'United-States'       # 压倒性多数
    }


    for col, default_val in defaults.items():
        info_df.loc[info_df[col] == 'unknown', col] = default_val
        # info_df[col] = info_df[col].astype(float)

    info_df['ID'] = np.arange(len(info_df)) + 1
    info_df = info_df[['ID'] + [c for c in info_df.columns if c != 'ID']]

    return info_df
