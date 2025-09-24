import json
import re
import pandas as pd
import sys
from inference_api import inference_gpt



def input_to_prompt(row, train_str, dataset_name, label_name):
    person_json = row.drop(label_name).to_dict()
    if dataset_name == "adult":
        person_json.pop("relationship", None) # REQ.1
    if dataset_name == "stroke":
        person_json.pop("id", None)
    person_json.pop("llm_prob_samples", None)
    json_str = json.dumps(person_json, ensure_ascii=False, indent=2)
    
    if train_str:
        if dataset_name == "adult":
            prompt = f"""You are an income prediction expert.
Estimate whether a person in 1994 would have an annual income greater than $50K, based on U.S. economic and social context of that year. Please use your own knowledge and the examples below as references.
Examples:
{train_str}
Now predict the following case:
Features: {json_str}
Question: How likely is this person to have income $50,000?
First, provide a short explanation of your reasoning.
Then answer withith one of: very unlikely, unlikely, somewhat unlikely, neutral, somewhat likely, likely, very likely. And provide explanation."""
        
        if dataset_name == "stroke":
            prompt = f"""You are a medical risk assessment expert.
Estimate wheter the following person will have a stroke. Please use your own knowledge and the examples below as references.
Examples:
{train_str}
Now predict the following case:
Features: {json_str}
Question: How likely is this patient to have a stroke?
First, provide a short explanation of your reasoning.
Then answer withith one of: very unlikely, unlikely, somewhat unlikely, neutral, somewhat likely, likely, very likely. And provide explanation."""
        
        if dataset_name == "heart":
            prompt = f"""You are a medical risk assessment expert.
Estimate wheter the following person will have a heart disease. Please use your own knowledge and the examples below as references.
Examples:
{train_str}
Now predict the following case:
Features: {json_str}
Question: How likely is this patient to have a heart disease?
First, provide a short explanation of your reasoning.
Then answer withith one of: very unlikely, unlikely, somewhat unlikely, neutral, somewhat likely, likely, very likely. And provide explanation."""
            
        if dataset_name == "loan":
            prompt = f"""You are a loan-risk analyst.
Estimate the probability that the following applicant will default on their loan. Please use your own knowledge and the examples below as references.
Examples:
{train_str}
Now predict the following case:
Features: {json_str}
Question: How likely is this applicant to be a defaulter?
First, provide a short explanation of your reasoning.
Then answer withith one of: very unlikely, unlikely, somewhat unlikely, neutral, somewhat likely, likely, very likely. And provide explanation."""
        
        
    else:
        if dataset_name == "adult":
            prompt = f"""You are an income prediction expert.
Estimate how likely a person in 1994 would have an annual income greater than $50K, based on U.S. economic and social context of that year.
Features: {json_str}
Question: How likely is this person to have income $50,000?
First, provide a short explanation of your reasoning.
Then answer withith one of: very unlikely, unlikely, somewhat unlikely, neutral, somewhat likely, likely, very likely. And provide explanation."""
        
        if dataset_name == "stroke":
            prompt = f"""You are a medical risk assessment expert.
Estimate how likely the following person will have a stroke.
Features: {json_str}
Question: How likely is this patient to have a stroke?
First, provide a short explanation of your reasoning.
Then answer withith one of: very unlikely, unlikely, somewhat unlikely, neutral, somewhat likely, likely, very likely. And provide explanation."""
        
        if dataset_name == "heart":
            prompt = f"""You are a medical risk assessment expert.
Estimate how likely the following person will have a heart disease.
Features: {json_str}
Question: How likely is this patient to have a heart disease?
First, provide a short explanation of your reasoning.
Then answer withith one of: very unlikely, unlikely, somewhat unlikely, neutral, somewhat likely, likely, very likely. And provide explanation."""
      
        if dataset_name == "loan":
            prompt = f"""You are a loan-risk analyst.
Estimate the probability that the following applicant will default on their loan.
Features: {json_str}
Question: How likely is this applicant to be a defaulter?
First, provide a short explanation of your reasoning.
Then answer withith one of: very unlikely, unlikely, somewhat unlikely, neutral, somewhat likely, likely, very likely. And provide explanation."""
        
        
    return prompt

def train_to_prompt(train_df, dataset_name, label_name, max_samples=None):
    train_str = ""
    for idx, row in train_df.iterrows():
        if max_samples is not None and idx >= max_samples:
            break
        person_json = row.drop(label_name).to_dict()
        if dataset_name == "adult":
            person_json.pop("relationship", None) # REQ.1
        json_str = json.dumps(person_json, ensure_ascii=False, indent=2)
        train_str += f"- Features: {json_str}\n"
        
        # ( 
        #         "You are an income prediction expert. " 
        #         "Estimate whether the following person has an annual income greater than $50K.\n\n" 
        #         "This person lived in 1994; please base your judgment on the U.S. economic and social context of that year. "
        #         f"Person information (in JSON):\n{json_str}\n\n" "Does this person have income >50K?\n" 
        #         "First, provide a short explanation of your reasoning.
Then answer withith one of: yes, no. " )

        # (
        #     "You are a medical risk assessment expert. Estimate the probability that the following person will have a stroke.\n\n"
        #     f"Person information:\n{json_str}\n\n"
        #     "How likely is this patient to have a stroke?\n"
        #     "First, provide a short explanation of your reasoning.
Then answer withith one of: very unlikely, unlikely, somewhat unlikely, neutral, somewhat likely, likely, very likely. And provide explanation"
        # )

        label = row[label_name]
        if label > 0.5:
            train_str += f"  Label: yes\n"
        else:
            train_str += f"  Label: no\n"
    return train_str
            
cat_to_num = {
    "very unlikely": 0.05,
    "unlikely": 0.2,
    "somewhat unlikely": 0.35,
    "neutral": 0.5,
    "somewhat likely": 0.65,
    "likely": 0.8,
    "very likely": 0.95
}

def extract_label(text):
    label_out = "neutral"
    for label in ["very unlikely", "somewhat unlikely", "unlikely", "neutral", "somewhat likely", "very likely", "likely"]:
        if re.search(rf"\b{re.escape(label)}\b", text, re.IGNORECASE):
            label_out = label.lower()
            break
    label_out = cat_to_num.get(label_out, 0.5)
    return label_out

def extract_label_loan(paragraph):
    labels = ["very unlikely", "very likely", "somewhat unlikely", "somewhat likely", "unlikely", "likely", "neutral"]
    min_index = len(paragraph) + 1
    first_label = None
    for label in labels:
        idx = paragraph.lower().find(label.lower())
        if idx != -1 and idx < min_index:
            min_index = idx
            first_label = label
    return first_label

def main(dataset_name, test_dfpath, train_dfpath, label_name, model, save_path, early_stop = None, max_samples=None):
    df = pd.read_csv(test_dfpath)
    df["llm_prob_samples"] = None

    train_df = pd.read_csv(train_dfpath)
    train_str = train_to_prompt(train_df, dataset_name, label_name, max_samples)
    # ## temp
    # with open('results/icl_0_stroke.json', "r", encoding="utf-8") as f:
    #     icl_0_data = json.load(f)
    # with open('results/icl_5_stroke.json', "r", encoding="utf-8") as f:
    #     icl_5_data = json.load(f)
    # ##
    # for idx, row in df.iterrows():
    #     # ## tmp
    #     # if icl_0_data[idx]['llm_prob_samples'] == icl_5_data[idx]['llm_prob_samples']:
    #     #     continue
    #     # ##
    #     if early_stop is not None and idx >= early_stop:
    #         break
    #     response_str = inference_gpt(messages=[{"role": "user", "content": input_to_prompt(row, train_str, dataset_name, label_name)}], verbose=(idx < 1), model=model)
    #     df.at[idx, "llm_prob_samples"] = extract_label(response_str)

    messages = []
    for idx, row in df.iterrows():
        # ## tmp
        # if icl_0_data[idx]['llm_prob_samples'] == icl_5_data[idx]['llm_prob_samples']:
        #     continue
        # ##
        if early_stop is not None and idx >= early_stop:
            break
        messages.append([{"role": "user", "content": input_to_prompt(row, train_str, dataset_name, label_name)}])
    response_strs = inference_gpt(messages=messages, model=model, verbose=True)
    for idx, response_str in enumerate(response_strs):
        if dataset_name == "loan":
            label = extract_label_loan(response_str)
        else:
            label = extract_label(response_str)
        df.at[idx, "llm_prob_samples"] = cat_to_num.get(label, 0.5)

    df.to_json(save_path, orient="records", indent=2, force_ascii=False)

if __name__ == "__main__":
    # for max_samples in [0]:
    #     print(f"\n\n\n[icl_{max_samples}_stroke]\n\n")
    #     main("stroke", "stroke_balanced_300.csv", "stroke_train_10.csv", "stroke", "gpt-4.1-mini", f"case/icl_{max_samples}_stroke.json", early_stop=50, max_samples=max_samples)
    
    # print("\n\n\n[icl_0_stroke]\n\n")
    # main("stroke", "stroke_balanced_300.csv", "stroke_train_5.csv", "stroke", "gpt-4.1-mini", "results/icl_0_stroke.json", max_samples=0)
    # print("\n\n\n[icl_5_stroke]\n\n")
    # main("stroke", "stroke_balanced_300.csv", "stroke_train_5.csv", "stroke", "gpt-4.1-mini", "results/icl_5_stroke.json")
    # print("\n\n\n[icl_10_stroke]\n\n")
    # main("stroke", "stroke_balanced_300.csv", "stroke_train_10.csv", "stroke", "gpt-4.1-mini", "results/icl_10_stroke.json")
    # print("\n\n\n[icl_0_adult]\n\n")
    # main("adult", "adult_processed_300.csv", "adult_train_5.csv", "label", "gpt-4.1-mini", "results/icl_0_adult.json", max_samples=0)
    # print("\n\n\n[icl_5_adult]\n\n")
    # main("adult", "adult_processed_300.csv", "adult_train_5.csv", "label", "gpt-4.1-mini", "results/icl_5_adult.json")
    # print("\n\n\n[icl_10_adult]\n\n")
    # main("adult", "adult_processed_300.csv", "adult_train_10.csv", "label", "gpt-4.1-mini", "results/icl_10_adult.json")

    # model = 'gpt-4.1-mini'
    model = 'gemini-2.5-pro'

    # test
    # for dataset_name, data_path, label in [("stroke", "stroke_balanced_300.csv", "stroke")]:
    #     main(dataset_name, data_path, f"{dataset_name}_train_5.csv", label, model, f"gemini_results/icl_5_{dataset_name}.json", early_stop=2)

    for dataset_name, data_path, label in [("loan", "loan_300.csv", "loan_status")]: # ("stroke", "stroke_balanced_300.csv", "stroke"), ("adult", "adult_processed_300.csv", "label"), ("heart", "heart_failure.csv", "Heart Disease")]:
        print(f"\n\n\n[icl_5_{dataset_name}]\n\n")
        main(dataset_name, data_path, f"{dataset_name}_train_5.csv", label, model, f"gemini_results/icl_5_{dataset_name}_1.json")
        print(f"\n\n\n[icl_10_{dataset_name}]\n\n")
        main(dataset_name, data_path, f"{dataset_name}_train_10.csv", label, model, f"gemini_results/icl_10_{dataset_name}_1.json")

