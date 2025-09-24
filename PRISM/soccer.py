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
    api_key = 'Your api key'
)

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
    \"\"\"{report_text}\"\"\"
    Based on the report, briefly comment on the situation of this match, covering the following aspects:
    1. Squad Quality. 2. Head-to-head records. 3. Recent form. 4. Player availability and fitness. 5. External conditions. Provide the results following the format: 1) XXX \n\n2) XXX \n\n3) XXX \n\n4) XXX \n\n5) XXX
    """

    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": user_prompt}],
    ).choices[0].message.content.strip()

    factor_list = []
    for i in range(1,10):
        for ll in response.split('\n\n'):
            if str(i) + ') ' in ll:
                ll = ll.split(str(i) + ') ')[1]
                factor_list.append(ll)

    results_json = []
    feature_impacts = {}
    total_score = 0.0

    for idx, tg_factor in enumerate(factor_list):
        print(tg_factor)
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
                        f"We have information of a football match between {teams} at {hometeam}'s home.\n\n{baseline}\n\n"
                        f"Q1. Estimate the likelihood (1-10) that the home team will win the match.\n"
                        f"- 1 = very unlikely, 10 = very likely.\n\n"
                        f"Q2. New info: {tg_factor}\n\n"
                        f"If the new info directly influences the estimation, give a new score. If no, provide the original estimation.\n"
                        f"Briefly explain and provide the final estimations in the format ##Final Result##: @Q1:[1-10] Q2:[1-10]@"
                    )

                    resp = client.chat.completions.create(
                        model="gpt-5-mini",
                        messages=[{"role": "user", "content": user_prompt}],
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

    final_prob = logit_to_prob(total_score)
    print('final probability', final_prob)