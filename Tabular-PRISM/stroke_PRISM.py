import numpy as np
import json
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import argparse
from openai import OpenAI
import random
from utils_stroke import *
from sklearn.metrics import brier_score_loss
import sys
sys.stdout = open("shap_stroke_sc_gpt41_third.txt", "w", encoding="utf-8")

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
df = pd.read_csv('data_process/stroke_balanced_300.csv')

df.drop('id', axis=1, inplace=True)
df.dropna(inplace=True)

df['gender'] = (df['gender']== 'Male').astype(int)
df['ever_married'] = (df['ever_married']=='Yes').astype(int)
df['Residence_type'] = (df['Residence_type']== 'Urban').astype(int)

df = df.rename(columns={"avg_glucose_level": "average glucose level"})
df = df.rename(columns={"ever_married": "marital status"})
df = df.rename(columns={"work_type": "work type"})
df = df.rename(columns={"heart_disease": "heart disease"})
df = df.rename(columns={"smoking_status": "smoking status"})
df = df.rename(columns={"Residence_type": "residence type"})
df = df.rename(columns={"bmi": "Body Mass Index (BMI)"})

X = df.drop(['stroke'], axis = 1)
y = df['stroke']
size = len(X)
columns = X.columns.to_list()

all_pred = []
all_label = []
results_json = [] 

for i in range(len(X)):
    print(f"=== Sample f{i} ===")
    info = X.iloc[i]
    print(info, y.iloc[i])

    base_logit = prob_to_logit(0.05)
    score = 0
    scores = []
    feature_impacts = {} 

    for ff in columns:
        tg_feature = ff
        tg_value = info[tg_feature]
        impacts = []
        info_df = []
        numbers = []

        for j in range(10):
            bg_feature = random_subset_at_least_k(columns.copy(), tg_feature)
            bg_info = info[bg_feature]

            info2 = info.copy()
            for cc in columns:
                if cc not in bg_feature:
                    info2[cc] = 'unknown'
            info2['mark'] = str(j) + 'A'

            info3 = info2.copy()
            info3[tg_feature] = tg_value
            info3['mark'] = str(j) + 'B'

            if random.choice([1,2]) == 1:
                info_df.append(info2)
                info_df.append(info3)
            else:
                info_df.append(info3)
                info_df.append(info2)
        
        info_df = pd.DataFrame(info_df)
        info_df = info_df.astype('str')
        info_df = process_df(info_df)

        if info_df.iloc[0][tg_feature] == info_df.iloc[1][tg_feature]:
            impact = 0
            print(ff, impact)
        else:
            while True:
                try:
                    info_df_q = info_df.drop(columns= 'mark', axis = 1)
                    duplicates = info_df.drop(columns= 'mark', axis = 1)[info_df.drop(columns= 'mark', axis = 1).duplicated(keep=False)]
                    info_df_q = info_df_q.to_markdown(index= False)
                    print(info_df_q)
                    user_prompt = info_df_q + \
                    '\n\nYour task is to evaluate the likelihood that each patient has stroke.\n\n' + \
                    ('For each patient:\n'
                    '- Conduct a brief analysis.\n'
                    '- Give a risk score from 1 to 10 (1 = very unlikely, 10 = very likely).\n\n'
                    'Provide the final result in the format ##Final Result##: @ID1: 1-10; @ID2: 1-10....')
                    
                    response = client.chat.completions.create(
                        model="gpt-4.1-mini", 
                        messages=[ {"role": "user", "content": user_prompt}]
                    ).choices[0].message.content.strip()
                    print('LLM Response: ' + response)                     
                    if 'Final Result##' in response:
                        resp = response.split('Final Result##')[1]
                    else:
                        resp = response.split('Final Result')[1]
                    
                    numbers = re.findall(r":\s*(\d+)", resp)
                    numbers = [int(j) for j in numbers]
                    info_df['position'] = numbers 

                    impact_list = []
                    impact_pair_list = []
                    for mark in range(int(len(info_df)/2)):
                        pos1 = info_df.loc[info_df['mark'] == str(mark) + 'A','position'].values[0]
                        pos2 = info_df.loc[info_df['mark'] == str(mark) + 'B','position'].values[0]

                        logit1 = prob_to_logit(pos1/10 - 0.05)
                        logit2 = prob_to_logit(pos2/10 - 0.05)
                        
                        impact_list.append(logit2 - logit1)
                        impact_pair_list.append((pos1, pos2))


                    impact = np.mean(impact_list) 
                    print(ff, round(impact, 3), impact_pair_list)
                    break
                except Exception as e:
                    print('.. error happens')
                    print(e)
                    pass
        score = score + impact

        feature_impacts[ff] = {
            "impact": float(impact),
            "numbers": numbers
        }

    print('***initial score and prob', round(score,3), round(logit_to_prob(score + base_logit),3))
    all_pred.append(round(logit_to_prob(score + base_logit),3))
    all_label.append(y.iloc[i])

    final_score = score
    final_prob = logit_to_prob(score + base_logit)

    results_json.append({
        "sample_id": i,
        "true_label": int(y.iloc[i]),
        "final_score": float(final_score),
        "final_prob": float(final_prob),
        "features": X.iloc[i].to_dict(),
        "feature_impacts": feature_impacts
    })
    
    if i > 10:
        all_pred2 = np.array(all_pred)
        all_label2 = np.array(all_label)

        auroc = roc_auc_score(y_true=all_label2, y_score= all_pred2)
        b_score = brier_score_loss(all_label2, all_pred2)

        print('overall', auroc, 'brier', b_score, 'in total ', i + 1)
    
    with open("results/shap/shap_stroke_sc_gpt41_third.json", "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    
    print('.................')
