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

df = pd.read_csv("data_process/adult_300.csv")
df = df.rename(columns={"education": "education level"})
df = df.rename(columns={"hours.per.week": "working hours per week"})
df = df.rename(columns={"marital.status": "marital status"})
df = df.rename(columns={"native.country": "native country"})
df = df.rename(columns={"capital.gain": "capital gain"})
df = df.rename(columns={"capital.loss": "capital loss"})
df = process_df(df)

df.dropna(inplace=True)

X = df.drop(['income', 'sex', 'race', 'fnlwgt', 'education.num', 'relationship'], axis=1)
y = (df['income'] == '>50K').astype(int)
if "llm_prob_samples" not in df.columns:
    df["llm_prob_samples"] = None
if "llm_prob_final" not in df.columns:
    df["llm_prob_final"] = None
df = df.astype({"llm_prob_samples": "object", "llm_prob_final": "object"})


def transform_row(row):
    label = 1 if row["income"] == ">50K" else 0

    original_features = {k: row[k] for k in row.index 
                         if k not in ["income", "llm_prob_samples", "llm_prob_final", 
                                      "fnlwgt", "education.num", "relationship"]}

    return {
        "original_features": original_features,
        "label": label,
        "llm_prob_samples": row.get("llm_prob_samples", {}),
        "llm_prob_final": row.get("llm_prob_final", None)
    }



for idx, row in df.iterrows():
    print(f"Processing {idx}/{len(df)}")
    
    sample_probs = []
    person_json = row.drop("income").to_dict()
    person_json.pop("llm_prob_samples", None)
    person_json.pop("llm_prob_final", None)
    json_str = json.dumps(person_json, ensure_ascii=False, indent=2)

    pos_probs = []
    prompt_income = ( 
        "You are an income prediction expert. " 
        "Estimate the probability that the following person has an annual income greater than $50,000.\n\n" 
        "This person lived in 1994; please base your judgment on the U.S. economic and social context of that year. "
        f"Person information (in JSON):\n{json_str}\n\n" "How likely is this person to have income >$50,000?\n" 
        "First, provide a short explanation of your reasoning."
        "Then answer with one of: very unlikely, unlikely, somewhat unlikely, neutral, somewhat likely, likely, very likely. " 
        "And provide explanation." )
    for i in range(1):
        resp = inference_gpt(prompt_income)
        print("LLM Response (pos): " + resp)
        label = extract_label(resp)
        prob = cat_to_num.get(label, 0.5)
        pos_probs.append(prob)

    neg_probs = []
    prompt_income_neg = (
        "You are an income prediction expert. "
        "Estimate the probability that the following person has an annual income at most $50,000.\n\n"
        "This person lived in 1994; please base your judgment on the U.S. economic and social context of that year. "
        f"Person information (in JSON):\n{json_str}\n\n"
        "First, provide a short explanation of your reasoning."
        "Then answer with one of: very unlikely, unlikely, somewhat unlikely, neutral, somewhat likely, likely, very likely. "
        "And provide explanation."
    )
    print(prompt_income)
    for i in range(1):
        resp = inference_gpt(prompt_income_neg)
        print("LLM Response (neg): " + resp)
        label = extract_label(resp)
        prob = cat_to_num.get(label, 0.5)
        neg_probs.append(prob)

    prob_positive = sum(pos_probs) / len(pos_probs)
    prob_negative = sum(neg_probs) / len(neg_probs)
    final_prob = normalize_probs(prob_positive, prob_negative)

    df.at[idx, "llm_prob_samples"] = {"positive": pos_probs, "negative": neg_probs}
    df.at[idx, "llm_prob_final"] = final_prob

    records = [transform_row(row) for _, row in df.iterrows()]
    with open("results/contrast_adult_41.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)