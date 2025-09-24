import numpy as np
import json
import pandas as pd
import re
import random
import time
from openai import OpenAI
from sklearn.metrics import roc_auc_score, confusion_matrix, brier_score_loss

client = OpenAI(
    api_key="YOUR_API_KEY"
)

def inference_gpt(prompt: str, model: str = "gpt-4.1-mini", max_retries: int = 10) -> str:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=1,
                max_tokens=512,
            )
            return response.choices[0].message.content.strip().replace("\n", " ")
        except Exception as e:
            print(f"{e}, try {attempt+1}/{max_retries}")
            time.sleep(2)
    print("error")
    return "neutral"

def extract_label(text):
    for label in ["very unlikely", "somewhat unlikely", "unlikely", "neutral", "somewhat likely", "very likely", "likely"]:
        if re.search(rf"\b{re.escape(label)}\b", text, re.IGNORECASE):
            return label.lower()
    return "neutral"

cat_to_num = {
    "very unlikely": 0.05,
    "unlikely": 0.2,
    "somewhat unlikely": 0.35,
    "neutral": 0.5,
    "somewhat likely": 0.65,
    "likely": 0.8,
    "very likely": 0.95
}

def normalize_probs(prob_positive, prob_negative):
    total = prob_positive + prob_negative
    if total == 0:
        return 0.5
    return round(prob_positive / total, 3)

    
df = pd.read_csv('data_process/adult_300.csv')
df = df.rename(columns={"education": "education level"})
df = df.rename(columns={"hours.per.week": "working hours per week"})
df = df.rename(columns={"marital.status": "marital status"})
df = df.rename(columns={"native.country": "native country"})
df = df.rename(columns={"capital.gain": "capital gain"})
df = df.rename(columns={"capital.loss": "capital loss"})

df.dropna(inplace=True)

X = df.drop(['income', 'sex', 'race', 'fnlwgt', 'education.num', 'relationship'], axis=1)
y = (df['income'] == '>50K').astype(int)


n_samples = 10
X["llm_prob_samples"] = None

results_json = []

for idx in range(300):
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
            "How likely is this person to have income $50,000?\n"
            "First, provide a short explanation of your reasoning."
            "Then answer with one of: very unlikely, unlikely, somewhat unlikely, neutral, "
            "somewhat likely, likely, very likely. And provide explanation."
        )
        print(prompt_income)
        resp = inference_gpt(prompt_income)
        print(resp)
        label = extract_label(resp)
        prob = cat_to_num.get(label, 0.5)

        sample_probs.append(prob)
        print(f"Sample {i+1}: {prob} ({label})")

    X.at[idx, "llm_prob_samples"] = sample_probs
    results_json.append({
        "sample_id": idx,
        "features": person_json,
        "true_label": int(y.iloc[idx]),
        "probs": sample_probs
    })

    with open("results/sc_adult_41.json", "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
