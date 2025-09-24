import json
import random
import re
from openai import OpenAI

import sys
sys.stdout = open("code/bird/output_txt/synthetic_train_adult_41.txt", "w", encoding="utf-8")


def inference_gpt(prompt: str, model: str = "gpt-4.1-mini", max_retries: int = 10) -> str:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=1,
                max_tokens=1024,
            )
            return response.choices[0].message.content.strip().replace("\n", " ")
        except Exception as e:
            print(f"调用 API 失败: {e}, 重试 {attempt+1}/{max_retries}")
            time.sleep(2)
    print("多次调用失败，返回默认 'neutral'")
    return "neutral"

# 1. Load factor definitions
with open("code/bird/data_bird/factors_adult.json", "r", encoding="utf-8") as f:
    factors = json.load(f)


def extract_label(text):
    for label in ["very unlikely", "very likely", "somewhat unlikely", "somewhat likely", "unlikely", "likely", "neutral"]:
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

# def apply_constraints(fvals: dict) -> dict:
#     age = fvals.get("age")
#     work_type = fvals.get("work_type")

#     if age == "0-10":
#         fvals["work_type"] = "children"
#         fvals["smoking_status"] = random.choice(["Unknown", "never smoked"])

#     elif age == "10-20":
#         if work_type == "children":
#             fvals["smoking_status"] = random.choice(["Unknown", "never smoked"])

#     else:
#         if work_type == "children":
#             valid_work_types = [w for w in factors["work_type"] if w != "children"]
#             fvals["work_type"] = random.choice(valid_work_types)

#     return fvals

import random

# 假设 factor 取值列表都存在于这个字典里
factors = {
    "age": ["15-20","20-30","30-40","40-50","50-60","60-70","70-83"],
    "workclass": ["?","Federal-gov","Local-gov","Private","Self-emp-inc","Self-emp-not-inc","State-gov"],
    "education": ["10th","11th","12th","5th-6th","9th","Assoc-acdm","Assoc-voc","Bachelors","Doctorate","HS-grad","Masters","Prof-school","Some-college"],
    "marital.status": ["Divorced","Married-civ-spouse","Married-spouse-absent","Never-married","Separated","Widowed"],
    "occupation": ["?","Adm-clerical","Craft-repair","Exec-managerial","Farming-fishing","Handlers-cleaners","Machine-op-inspct","Other-service","Priv-house-serv","Prof-specialty","Protective-serv","Sales","Tech-support","Transport-moving"],
    "capital.gain": ["0","1-1000","1000-5000","5000-10000","10000-20000","20000-30000"],
    "capital.loss": ["0","1-500","500-1000","1000-2000","2000-3000"],
    "hours.per.week": ["0-20","20-30","30-40","40-50","50-60","60-80","80-90"],
    "native.country": ["?","Canada","Dominican-Republic","England","Guatemala","Italy","Jamaica","Japan","Mexico","Philippines","Taiwan","Trinadad&Tobago","United-States","Vietnam"]
}

def apply_constraints(fvals: dict) -> dict:
    age = fvals.get("age")
    workclass = fvals.get("workclass")
    education = fvals.get("education")
    occupation = fvals.get("occupation")
    gain = fvals.get("capital.gain")
    loss = fvals.get("capital.loss")
    hours = fvals.get("hours.per.week")

    if gain != "0" and loss != "0":
        if random.random() < 0.5:
            fvals["capital.loss"] = "0"
        else:
            fvals["capital.gain"] = "0"

    if age == "15-20" and education in {"Bachelors","Masters","Prof-school","Doctorate","Assoc-acdm","Assoc-voc"}:
        valid_edus = [e for e in factors["education"] if e in {"10th","11th","12th","9th","5th-6th","HS-grad","Some-college"}]
        fvals["education"] = random.choice(valid_edus)
    if age == "20-30" and education in {"Prof-school","Doctorate"}:
        valid_edus = ["Bachelors","Masters","Some-college"]
        fvals["education"] = random.choice(valid_edus)

    if occupation == "Priv-house-serv" and workclass in {"Federal-gov","State-gov","Local-gov"}:
        fvals["workclass"] = "Private"
    if workclass in {"Federal-gov","State-gov","Local-gov"} and occupation in {"Self-emp-inc","Self-emp-not-inc"}:
        fvals["occupation"] = random.choice([o for o in factors["occupation"] if o not in {"Self-emp-inc","Self-emp-not-inc"}])

    if age == "15-20" and hours == "80-90":
        fvals["hours.per.week"] = random.choice(["20-30","30-40"])
    if hours == "0-20" and occupation in {"Exec-managerial","Prof-specialty","Transport-moving"}:
        fvals["hours.per.week"] = random.choice(["40-50","50-60"])

    return fvals

results = []
for i in range(300):
    print(f"=== sample {i} ====")
    fvals = {factor: random.choice(bins) for factor, bins in factors.items()}
    fvals = apply_constraints(fvals) 
    print(fvals)
    conds = ", ".join(f"{factor} = {val}" for factor, val in fvals.items())
    
    # prompt_app = (
    #     f"You are a medical risk assessment expert. "
    #     f"Given the following person information: {conds}, how likely is the person to have a stroke? "
    #     "Answer with one of: very unlikely, unlikely, somewhat unlikely, neutral, "
    #     "somewhat likely, likely, very likely."
    # )
    
    # prompt_rej = (
    #     f"You are a medical risk assessment expert. "
    #     f"Given the following person information: {conds}, how likely is the person to NOT have a stroke? "
    #     "Answer with one of: very unlikely, unlikely, somewhat unlikely, neutral, "
    #     "somewhat likely, likely, very likely."
    # )
    prompt_app = (
        "You are an income prediction expert. " 
        "Estimate the probability that the following person has an annual income greater than $50,000.\n\n" 
        "This person lived in 1994; please base your judgment on the U.S. economic and social context of that year. "
        f"Given the following person information: {conds}, how likely is the person to have income >$50,000? "
        "Answer with one of: very unlikely, unlikely, somewhat unlikely, neutral, "
        "somewhat likely, likely, very likely."
    )
    
    prompt_rej = (
        "You are an income prediction expert. "
        "Estimate the probability that the following person has an annual income at most $50,000.\n\n"
        "This person lived in 1994; please base your judgment on the U.S. economic and social context of that year. "
        f"Given the following person information: {conds}, how likely is the person to have income <=$50,000? "
        "Answer with one of: very unlikely, unlikely, somewhat unlikely, neutral, "
        "somewhat likely, likely, very likely."
    )

    app_probs = []
    for _ in range(1):
        resp_app = inference_gpt(prompt_app)
        print("LLM Yes Response:" + resp_app)
        lab_app  = extract_label(resp_app)
        num_app  = cat_to_num.get(lab_app, 0.5)

        resp_rej = inference_gpt(prompt_rej)
        print("LLM No Response:" + resp_rej)
        lab_rej  = extract_label(resp_rej)
        num_rej  = cat_to_num.get(lab_rej, 0.5)

        total = num_app + num_rej
        if total > 0:
            app_probs.append(num_app / total)
        else:
            app_probs.append(0.5)

    label = sum(app_probs) / len(app_probs)

    results.append({
        "fvals": fvals,
        "prob": round(label, 3)
    })

# 5. Save results
    with open("code/bird/data_bird/synthetic_train_adult_41.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

print("Generated synthetic_dataset.json with 100 samples")
