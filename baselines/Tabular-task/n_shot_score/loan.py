import numpy as np
import json
import pandas as pd
import re
import random
import time
from openai import OpenAI
from sklearn.metrics import roc_auc_score, confusion_matrix, brier_score_loss
from llm_inference import *

def extract_prob(text):
    match = re.search(r"##Final Result##\s*:\s*([0-1](?:\.\d+)?|1\.0|0)", text)
    if match:
        prob = float(match.group(1))
        return max(0.0, min(1.0, prob))  
    return 0.5   

df = pd.read_csv('/groups/xuhan2/yangnan/Project/Decision/Shap/data_process/lending_300.csv')

X = df.drop(['loan_status'], axis=1)
y = df['loan_status']
size = len(X)

n_samples = 10
X["llm_prob_samples"] = None

results_json = []

for idx in range(size):
    row = X.iloc[idx]
    print(f"Processing {idx+1}/{len(X)}")

    sample_probs = []
    person_json = row.to_dict()
    person_json.pop("llm_prob_samples", None)
    json_str = json.dumps(person_json, ensure_ascii=False, indent=2)

    for i in range(n_samples):
        prompt_loan = (
            "You are a loan-risk analyst. Estimate the probability that the following applicant will default on their loan.\n\n"
            f"Person information (in JSON):\n{json_str}\n\n"
            "First, provide a short explanation of your reasoning.\n"
            "Then provide the final result strictly in the format:\n"
            "##Final Result##: <a single number between 0 and 1>\n"
        )
        print(prompt_loan)
        resp = call_llm('gpt41mini', prompt_loan)
        print(resp)
        prob = extract_prob(resp)

        sample_probs.append(prob)
        print(f"Sample {i+1}: {prob}")

    X.at[idx, "llm_prob_samples"] = sample_probs
    results_json.append({
        "sample_id": idx,
        "features": person_json,
        "true_label": int(y.iloc[idx]),
        "probs": sample_probs
    })

    with open("results/lending/41/10shot_value.json", "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
