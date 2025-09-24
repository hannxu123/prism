import json
import re

input_path = 'code/bird/data_bird/factors_adult.json'
with open(input_path, 'r', encoding='utf-8') as f:
    factors = json.load(f)

from openai import OpenAI
from llm_inference import *
# client = OpenAI(
#     api_key="REMOVEDproj-Bq_ITFLGCofcWOMgYAvCNuApNlFitM5xFnftVjrMShZ-iWzT4uUC81f8MrlKEaFkxKBYHevoxLT3BlbkFJUDX1ivcjCCkdQvAssTF9zdB4GyiVfWoe7zMF5F37Q-3RutEsFh1oycmQDaypjlBKrZxHrHo8EA"
# )

# client = OpenAI(
#     api_key="REMOVEDproj-bmA7_VLMa0F2_SpzaPNnY1z8rCUig53Rj0FE64TvGEkRKj3B3b_MoJaAjbvyN3hrqDgTNYUHPRT3BlbkFJqHdwiZZa50YKCrUN5C2V9XL4MsK_Sw636h2QevSYyi5nxHM0CkYUYMRblzu9w_3YfjqHB0xd0A"
# )

# def inference_gpt(prompt: str, model: str = "gpt-4.1-mini", max_retries: int = 10) -> str:
#     for attempt in range(max_retries):
#         try:
#             response = client.chat.completions.create(
#                 model=model,
#                 messages=[{"role": "user", "content": prompt}],
#                 temperature=1,
#                 max_tokens=1024,
#             )
#             return response.choices[0].message.content.strip().replace("\n", " ")
#         except Exception as e:
#             print(f"调用 API 失败: {e}, 重试 {attempt+1}/{max_retries}")
#             time.sleep(2)
#     print("多次调用失败，返回默认 'neutral'")
#     return "neutral"

# def extract_label(text):
#     for label in ["very unlikely", "very likely", "somewhat unlikely", "somewhat likely", "unlikely", "likely", "neutral"]:
#         if re.search(rf"\b{re.escape(label)}\b", text, re.IGNORECASE):
#             return label.lower()
#     return "neutral"

def extract_label(text):
    labels = ["very unlikely", "very likely", "somewhat unlikely", "somewhat likely", "unlikely", "likely", "neutral"]
    pattern = r"\b(" + "|".join(map(re.escape, labels)) + r")\b"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).lower()
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

cpt_init = {}

for factor, values in factors.items():
    cpt_init[factor] = {}
    for v in values:
        app_probs = []
        rej_probs = []

        # prompt_app = (
        #     f"You are a medical risk assessment expert. "
        #     f"Given that {factor} = {v}, how likely is the person to have a stroke? "
        #     "Answer with one of: very unlikely, unlikely, somewhat unlikely, neutral, "
        #     "somewhat likely, likely, very likely.\n\n"
        #     "Strictly follow this format:\n"
        #     "Label: <one of the options>\n"
        #     "Explanation: <your reasoning>"
        # )

        # prompt_rej = (
        #     f"You are a medical risk assessment expert. "
        #     f"Given that {factor} = {v}, how likely is the person NOT to have a stroke? "
        #     "Answer with one of: very unlikely, unlikely, somewhat unlikely, neutral, "
        #     "somewhat likely, likely, very likely.\n\n"
        #     "Strictly follow this format:\n"
        #     "Label: <one of the options>\n"
        #     "Explanation: <your reasoning>"
        # )
        prompt_app = (
            "You are an income prediction expert. "
            "Estimate the probability that the following person has an annual income greater than $50,000.\n\n"
            "This person lived in 1994; please base your judgment on the U.S. economic and social context of that year. "
            f"Given that {factor} = {v}, how likely is this person to have income $50,000?\n"
            "Answer with one of: very unlikely, unlikely, somewhat unlikely, neutral, "
            "somewhat likely, likely, very likely. And provide explanation."
        )
        prompt_rej = (
            "You are an income prediction expert. "
            "Estimate the probability that the following person has an annual income at most $50,000.\n\n"
            "This person lived in 1994; please base your judgment on the U.S. economic and social context of that year. "
            f"Given that {factor} = {v}, how likely is this person to have income <=$50,000?\n"
            "Answer with one of: very unlikely, unlikely, somewhat unlikely, neutral, "
            "somewhat likely, likely, very likely. And provide explanation."
        )

        for _ in range(1):
            # Approval
            resp_app = call_llm("gemini", prompt_app)
            label_app = extract_label(resp_app)
            num_app = cat_to_num.get(label_app, 0.5)
            # Rejection
            resp_rej = call_llm("gemini", prompt_rej)
            label_rej = extract_label(resp_rej)
            num_rej = cat_to_num.get(label_rej, 0.5)
            print(resp_app)
            print(label_app)
            print(resp_rej)
            print(label_rej)
            # Normalize this run
            total = num_app + num_rej
            if total > 0:
                p_app_run = num_app / total
                p_rej_run = num_rej / total
            else:
                p_app_run, p_rej_run = 0.5, 0.5
            
            app_probs.append(p_app_run)
            rej_probs.append(p_rej_run)
        
        # Average across runs
        avg_app = sum(app_probs) / len(app_probs)
        avg_rej = sum(rej_probs) / len(rej_probs)
        cpt_init[factor][v] = {
            "approve": round(avg_app, 3),
            "reject": round(avg_rej, 3)
        }

# 4. Save CPT
with open("code/bird/data_bird/cpt_init_adult_gemini.json", "w", encoding="utf-8") as f:
    json.dump(cpt_init, f, ensure_ascii=False, indent=2)
