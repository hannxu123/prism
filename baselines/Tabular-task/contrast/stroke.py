import json
import re
import pandas as pd
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

with open("data_process/stroke_balanced_300.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

df["llm_prob_stroke"] = None

for idx, row in df.iterrows():
    print(f"Processing {idx+1}/{len(df)}")
    json_str = row.to_dict
    prompt_stroke = (
        "You are a medical risk assessment expert. Estimate the probability that the following person  will have a stroke.\n\n"
        f"Person  information:\n{row['query']}\n\n"
        "First, provide a short explanation of your reasoning."
        "Then answer with one of: very unlikely, unlikely, somewhat unlikely, neutral, somewhat likely, likely, very likely. And provide explanation"
    )

    prompt_no_stroke = prompt_stroke.replace("have a stroke", "NOT have a stroke")

    resp_stroke = inference_gpt(prompt_stroke)
    label_stroke = extract_label(resp_stroke)
    prob_stroke = cat_to_num.get(label_stroke, 0.5)

    print(resp_stroke, label_stroke, prob_stroke)

    resp_no_stroke = inference_gpt(prompt_no_stroke)
    label_no_stroke = extract_label(resp_no_stroke)
    prob_no_stroke = cat_to_num.get(label_no_stroke, 0.5)

    print(resp_no_stroke, label_no_stroke, prob_no_stroke)

    final_prob = normalize_probs(prob_stroke, prob_no_stroke)
    df.loc[idx, "llm_prob_stroke"] = final_prob
    df.loc[idx, "prob_stroke_category"] = prob_stroke
    df.loc[idx, "prob_no_stroke_category"] = prob_no_stroke

    df.to_json("results/contrast_storke_41_0.json", orient="records", indent=2, force_ascii=False)