import numpy as np
import json
import pandas as pd
import re
import random
import time
from sklearn.metrics import roc_auc_score, confusion_matrix, brier_score_loss
from llm_inference import * 

def extract_prob(text):
    match = re.search(r"##Final Result##\s*:\s*([0-1](?:\.\d+)?|1\.0|0)", text)
    if match:
        prob = float(match.group(1))
        return max(0.0, min(1.0, prob))  # 裁剪到 [0,1]
    return 0.5   # fallback

df = pd.read_csv('/groups/xuhan2/jiahaowang/code/ICLR26/data/heart_failure.csv')
df.dropna(inplace=True)

X = df.drop(['Heart Disease'], axis=1)
y = df['Heart Disease'].astype(int)

n_samples = 10
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
        prompt_heart = (
            "You are a medical risk assessment expert. "
            "Estimate the probability that the following person will have heart disease.\n\n"
            f"Person information: {json_str}\n\n"
            "First, provide a short explanation of your reasoning.\n"
            "Then provide the final result strictly in the format:\n"
            "##Final Result##: <a single number between 0 and 1>\n"
        )

        print(prompt_heart)
        resp = call_llm('gpt41mini', prompt_heart)
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

    with open("/groups/xuhan2/yangnan/Project/Decision/Shap/results/heart/41/10shot_value_225to300_2.json", "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
