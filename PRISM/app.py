import json
import re
import fitz
from openai import OpenAI
import math 
import random
import numpy as np
import argparse
import time
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default = 100)
parser.add_argument('--name', type=str, default = 'Fuji')
args = parser.parse_args()
print('###############', args)

random.seed(args.seed)
np.random.seed(args.seed)

client = OpenAI(
    api_key = 'your api key'
)
pdf_path = "fruits/USAPPLE_OutlookReport_2024.pdf"


def random_subset(lst, tg_feature):
    lst2 = np.random.permutation(np.array(lst))
    idx = np.where(lst2 == tg_feature)[0][0]
    bg_list = lst2[0: idx]
    return bg_list.tolist()

def extract_pdf_text(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    return text

report_text = extract_pdf_text(pdf_path)

user_prompt = f"""
\"\"\"{report_text}\"\"\"
Based on the report, comment on the situation of U.S. conventional {args.name} apple, covering the following aspects:
1. 2025 production. 2. 2025 market demand. 3. 2025 storage. 4. 2025 Imports and exports. 5. 2025 Government policy. 6. 2025 costs. 7. 2025 varietal Competition. 
"""

response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[{"role": "user", "content": user_prompt}],
    temperature=0
).choices[0].message.content.strip()

factor_list = []
for i in range(1,10):
    for ll in response.split('\n\n#'):
        if  str(i) + '. *' in ll:
            ll = ll.split(str(i) + '. *')[1]
            factor_list.append(ll)

def prob_to_logit(p):
    epsilon = 1e-15
    p = max(epsilon, min(1 - epsilon, p))
    return math.log(p / (1 - p))

def logit_to_prob(logit):
    return 1 / (1 + math.exp(-logit))


results_json = []
feature_impacts = {}
total_score = 0.0

start = time.time()  # record start time
for idx, tg_factor in enumerate(factor_list):
    factor_key = f"factor_{idx}"
    replicates = []
    all_q = []
    for rep in range(5):  
        bg_subset = random_subset(factor_list, tg_factor) 

        baseline = "\n\n".join(bg_subset) if bg_subset else "no additional factor"
        test = baseline + "\n\n" + tg_factor if bg_subset else tg_factor
        while True:
            try:
                # prompt
                user_prompt = (
                    f"We have information about {args.name} apple this year.\n{baseline}\n\n"
                    f"Q1. Estimate the likelihood (1-10) that the price in 2024/2025 will be higher than this year.\n"
                    f"- 1 = very unlikely, 10 = very likely.\n\n"
                    f"Q2. New info: {tg_factor}\n"
                    f"If the new info directly influences the estimation, give a new score. If no, provide the original estimation.\n"
                    f"Briefly explain and provide the final estimations in the format ##Final Result##: @Q1:[1-10] Q2:[1-10]@"
                )

                resp = client.chat.completions.create(
                    model="gpt-4.1",
                    messages=[{"role": "user", "content": user_prompt}],
                    temperature=0.0
                ).choices[0].message.content.strip()

                resp = resp.split('Final Result')[-1].split('@')[1].replace('[', '').replace(']', '')
                q1 = int(resp.split('Q1:')[1].split(' Q2:')[0]) / 10 - 0.05
                q2 = int(resp.split('Q2:')[1]) / 10 - 0.05
                q = prob_to_logit(q2) - prob_to_logit(q1)
                all_q.append(q)
                break
            except:
                pass
    
    impact = np.mean(all_q)
    total_score = total_score + impact
    print(impact)
    print('.................')
end = time.time()  # record end time
print(f"Execution time: {end - start} seconds")

final_prob = logit_to_prob(total_score)
print('final probability', final_prob)