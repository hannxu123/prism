import json
import re
import pandas as pd
import time
import numpy as np
from llm_inference import * 
import sys

def extract_prob(text):
    match = re.search(r"##Final Result##\s*:\s*([0-1](?:\.\d+)?|1\.0|0)", text)
    if match:
        prob = float(match.group(1))
        return max(0.0, min(1.0, prob))  
    return 0.5   

df = pd.read_csv('/groups/xuhan2/yangnan/Project/Decision/Shap/data_process/stroke_balanced_300.csv')
df.drop('id', axis=1, inplace=True)
df.dropna(inplace=True)

df['gender'] = (df['gender'] == 'Male').astype(int)
df['ever_married'] = (df['ever_married'] == 'Yes').astype(int)
df['Residence_type'] = (df['Residence_type'] == 'Urban').astype(int)

df = df.rename(columns={
    "avg_glucose_level": "average glucose level",
    "ever_married": "marital status",
    "work_type": "work type",
    "heart_disease": "heart disease",
    "smoking_status": "smoking status",
    "Residence_type": "residence type",
    "bmi": "Body Mass Index (BMI)"
})


X = df.drop(['stroke'], axis=1)
y = df['stroke']

n_samples = 5
X["llm_prob_samples"] = None

results_json = []

for idx in range(1):
    row = X.iloc[idx]
    print(f"Processing {idx+1}/{len(X)}")

    sample_probs = []
    raw_outputs = []

    person_json = row.to_dict()
    person_json.pop("llm_prob_samples", None)
    json_str = json.dumps(person_json, ensure_ascii=False, indent=2)

    for i in range(n_samples):
        prompt_stroke = (
            "You are a medical risk assessment expert. "
            "Estimate the probability that the following person will have a stroke.\n\n"
            f"Person information:\n{json_str}\n\n"
            "First, provide a short explanation of your reasoning.\n"
            "Then provide the final result strictly in the format:\n"
            "##Final Result##: <a single number between 0 and 1>\n"
        )

        print(prompt_stroke)
        resp = call_llm('gpt41mini', prompt_stroke)
        print(resp)
        prob = extract_prob(resp)
        sample_probs.append(prob)
        raw_outputs.append(resp)

        print(f"Sample {i+1}: {prob}")

    X.at[idx, "llm_prob_samples"] = sample_probs
    results_json.append({
        "sample_id": idx,
        "features": person_json,
        "true_label": int(y.iloc[idx]),
        "probs": np.mean(sample_probs),
    })

    with open("/groups/xuhan2/yangnan/Project/Decision/Shap/results/stroke/41/useless.json", "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
