import numpy as np
import json
import pandas as pd
import re
import random
import time
from openai import OpenAI

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


df = pd.read_csv('/groups/xuhan2/yangnan/Project/Decision/Shap/data_process/lending_300.csv') # 


X = df.drop(['loan_status'], axis = 1)
y = df['loan_status'] 

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
        prompt_loan = (
            "You are a loan-risk analyst. Estimate the probability that the following applicant will default on their loan.\n\n"
            f"Person information (in JSON):\n{json_str}\n\n"
            "How likely is this applicant to be a defaulter?\n"
            "AFirst, provide a short explanation of your reasoning."
            "Then answer with one of: very unlikely, unlikely, somewhat unlikely, neutral, somewhat likely, likely, very likely. And provide explanation"
        )
        print(prompt_loan)
        resp = inference_gpt(prompt_loan)
        print(resp)
        label = extract_label(resp)
        prob = cat_to_num.get(label, 0.5)

        sample_probs.append(prob)
        print(f"Sample {i+1}: {prob} ({label})")

    # 保存结果
    X.at[idx, "llm_prob_samples"] = sample_probs
    results_json.append({
        "sample_id": idx,
        "features": person_json,
        "true_label": int(y.iloc[idx]),
        "probs": sample_probs
    })

    with open("results/sc/sc_loan_41_192to230.json", "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
