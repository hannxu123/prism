import json
import re
import pandas as pd
from openai import OpenAI
import sys
import numpy as np
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

df = pd.read_csv("./data/heart_failure.csv")
df.dropna(inplace=True)

X = df.drop(['Heart Disease'], axis=1)
y = (df['Heart Disease']).astype(int)

results_json = []

for idx, row in X.iterrows():
    print(f"Processing {idx+1}/{len(df)}")
    
    sample_probs = []
    person_json = row.to_dict()
    person_json.pop("llm_prob_samples_positive", None)
    person_json.pop("llm_prob_samples_negative", None)
    person_json.pop("llm_prob_final", None)
    json_str = json.dumps(person_json, ensure_ascii=False, indent=2)

    pos_probs = []
    prompt_loan = ( 
        "You are a medical risk assessment expert.Estimate the probability that the following person will have heart disease.\n\n"
        f"Person information: {json_str}\n"
        "How likely is this patient to have heart disease?\n"
        "First, provide a short explanation of your reasoning."
        "Then answer with one of: very unlikely, unlikely, somewhat unlikely, neutral, somewhat likely, likely, very likely. And provide explanation."
        )
    for i in range(1):
        resp = inference_gpt(prompt_loan)
        print("LLM Response (pos): " + resp)
        label = extract_label(resp)
        prob = cat_to_num.get(label, 0.5)
        pos_probs.append(prob)

    neg_probs = []
    prompt_loan_neg = (
        "You are a medical risk assessment expert.Estimate the probability that the following person will NOT have heart disease.\n\n"
        f"Person information: {json_str}\n"
        "How likely is this patient to NOT have heart disease?\n"
        "First, provide a short explanation of your reasoning."
        "Then answer with one of: very unlikely, unlikely, somewhat unlikely, neutral, somewhat likely, likely, very likely. And provide explanation."
        )
    print(prompt_loan)
    for i in range(1):
        resp = inference_gpt(prompt_loan_neg)
        print("LLM Response (neg): " + resp)
        label = extract_label(resp)
        prob = cat_to_num.get(label, 0.5)
        neg_probs.append(prob)

    prob_positive = sum(pos_probs) / len(pos_probs)
    prob_negative = sum(neg_probs) / len(neg_probs)
    final_prob = normalize_probs(prob_positive, prob_negative)

    results_json.append({
        "sample_id": idx,
        "features": person_json,
        "true_label": int(y.iloc[idx]),
        "llm_prob_samples_positive": pos_probs,
        "llm_prob_samples_negative": neg_probs,
        "llm_prob_final": final_prob
    })

    with open("./results/contrast_heart_41.json", "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)