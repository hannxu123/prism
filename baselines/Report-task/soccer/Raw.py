import json
import re
import fitz
from openai import OpenAI
import math 
import random
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default = 100)
args = parser.parse_args()
print('###############', args)

random.seed(args.seed)
np.random.seed(args.seed)

client = OpenAI(
    api_key = 'REMOVEDproj-bmA7_VLMa0F2_SpzaPNnY1z8rCUig53Rj0FE64TvGEkRKj3B3b_MoJaAjbvyN3hrqDgTNYUHPRT3BlbkFJqHdwiZZa50YKCrUN5C2V9XL4MsK_Sw636h2QevSYyi5nxHM0CkYUYMRblzu9w_3YfjqHB0xd0A'
)

def extract_first_number(s):
    match = re.search(r"\d+(\.\d+)?", s)
    if match:
        return float(match.group())
    return None

with open("soccer.txt", "r") as f:
    content = f.read()

def prob_to_logit(p):
    epsilon = 1e-15
    p = max(epsilon, min(1 - epsilon, p))
    return math.log(p / (1 - p))

def logit_to_prob(logit):
    return 1 / (1 + math.exp(-logit))

def random_subset(lst, tg_feature):
    lst2 = np.random.permutation(np.array(lst))
    idx = np.where(lst2 == tg_feature)[0][0]
    bg_list = lst2[0: idx]
    return bg_list.tolist()

for report in content.split('@@@@@')[1:]:
    teams = report.split('##base score')[0].replace('vs', 'and').replace('\n', '')
    hometeam = report.split('##base score')[0].split('vs')[0]

    print(teams)
    report_text = report.split('##\n\n')[1]

    user_prompt = f"""
    We have information of a football match between {teams} at {hometeam}'s home. "
    \"\"\"{report_text}\"\"\"
    Based on the report, estimate how likely the home team will win the match. Briefly analyze and provide your answer in: @@Final probability: [0-1]@@. 
    """

    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": user_prompt}],
    ).choices[0].message.content.strip()

    resp = response.split('@@')[1]
    print(resp)
    print(float(extract_first_number(resp)))

