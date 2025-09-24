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
    api_key = 'REMOVEDproj-x-xeXFQ8c7DaXZX8PFjX5qk-70vvuA5JaQPDb768662UnxQCDoYS4bL6Pi6GOdFMgQj4HNpdwST3BlbkFJiL5U4r7bcAh1jSirjA8dAuzPJuKFy7ViOnNjDyY4PlqIZLtmfnQt7NifxM5jEBXLdTP1KDZpUA'
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
Based on the report, estimate how likely the price of U.S. conventional {args.name} apple will increase in the year 2024/2025. Briefly analyze and provide your answer in: @@Final Result: [0-1]@@. 
"""

prob = []
for i in range(10):
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": user_prompt}],
    ).choices[0].message.content.strip()

    resp = response.split('@@')[1]
    print(float(extract_first_number(resp)))
    prob.append(float(extract_first_number(resp)))

print('final result', np.mean(prob))