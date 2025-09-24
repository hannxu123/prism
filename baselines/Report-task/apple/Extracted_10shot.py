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
parser.add_argument('--name', type=str, default = 'Fuji')
args = parser.parse_args()
print('###############', args)

def extract_first_number(s):
    match = re.search(r"\d+(\.\d+)?", s)
    if match:
        return float(match.group())
    return None

random.seed(args.seed)
np.random.seed(args.seed)

client = OpenAI(
    api_key = 'REMOVEDproj-bmA7_VLMa0F2_SpzaPNnY1z8rCUig53Rj0FE64TvGEkRKj3B3b_MoJaAjbvyN3hrqDgTNYUHPRT3BlbkFJqHdwiZZa50YKCrUN5C2V9XL4MsK_Sw636h2QevSYyi5nxHM0CkYUYMRblzu9w_3YfjqHB0xd0A'
)
pdf_path = "fruits/USAPPLE_OutlookReport_2024.pdf"

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


prob = []
for i in range(10):

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": user_prompt}],
        temperature=0
    ).choices[0].message.content.strip()

    user_prompt2 = f"""
    \"\"\"{response}\"\"\"
    Based on the report, estimate how likely the price of U.S. conventional {args.name} apple will increase in the year 2024/2025. Briefly analyze and provide your answer in: @@Final Result: [0-1]@@. 
    """

    response2 = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": user_prompt2}],
    ).choices[0].message.content.strip()

    resp = response2.split('@@')[1]
    print(float(extract_first_number(resp)))
    prob.append(float(extract_first_number(resp)))

print('final result', np.mean(prob))

