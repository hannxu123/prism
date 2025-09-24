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

df = pd.read_csv('data_process/adult_300.csv')
df = df.rename(columns={"education": "education level"})
df = df.rename(columns={"hours.per.week": "working hours per week"})
df = df.rename(columns={"marital.status": "marital status"})
df = df.rename(columns={"native.country": "native country"})
df = df.rename(columns={"capital.gain": "capital gain"})
df = df.rename(columns={"capital.loss": "capital loss"})


X = df.drop(['income', 'sex', 'race', 'fnlwgt', 'education.num', 'relationship'], axis=1)
y = (df['income'] == '>50K').astype(int)

n_samples = 1
X["llm_prob_samples"] = None

results_json = []

for idx in range(len(X)):
    row = X.iloc[idx]
    print(f"Processing {idx+1}/{len(X)}")

    sample_probs = []
    person_json = row.to_dict()
    person_json.pop("llm_prob_samples", None)
    json_str = json.dumps(person_json, ensure_ascii=False, indent=2)

    for i in range(n_samples):
        prompt_income = (
            "You are an income prediction expert. "
            "Estimate the probability that the following person has an annual income greater than $50,000.\n\n"
            "This person lived in 1994; please base your judgment on the U.S. economic and social context of that year. "
            f"Person information (in JSON):\n{json_str}\n\n"
            "First, provide a short explanation of your reasoning.\n"
            "Then provide the final result strictly in the format:\n"
            "##Final Result##: <a single number between 0 and 1>\n"
        )
        print(prompt_income)
        resp = call_llm('gpt41mini', prompt_income)
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

    with open("results/adult/41/1shot_value_yes.json", "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
