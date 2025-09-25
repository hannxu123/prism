import pandas as pd
import json

input_file = "data_process/stroke_balanced_300.csv"    
output_file = "code/bird/data_bird/stroke_balanced_300_binned.csv" 
json_file = "code/bird/data_bird/factors_stroke.json" 

df = pd.read_csv(input_file)

# ---------------------------
# ---------------------------

# 1. Age bins
age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 83]  
age_labels = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-83"]

df["age"] = pd.cut(df["age"], bins=age_bins, labels=age_labels, include_lowest=True, right=False)

# 2. Avg_glucose_level bins
glucose_bins = [55, 80, 100, 125, 160, 200, 240, 273]
glucose_labels = ["55-80", "80-100", "100-125", "125-160", "160-200", "200-240", "240-273"]

df["avg_glucose_level"] = pd.cut(df["avg_glucose_level"], bins=glucose_bins,
                                 labels=glucose_labels, include_lowest=True, right=False)

# 3. BMI bins
bmi_bins = [0, 18.5, 24.9, 29.9, 34.9, 39.9, 49.9, 56.7]
bmi_labels = ["<18.5", "18.5-24.9", "25-29.9", "30-34.9", "35-39.9", "40-49.9", "50-56.7"]

df["bmi"] = pd.cut(df["bmi"], bins=bmi_bins, labels=bmi_labels, include_lowest=True, right=False)

# 4. Hypertension, Heart_disease → Yes/No
df["hypertension"] = df["hypertension"].map({0: "No", 1: "Yes"})
df["heart_disease"] = df["heart_disease"].map({0: "No", 1: "Yes"})

df.to_csv(output_file, index=False, encoding="utf-8")
print(f" {output_file}")

category_dict = {}

for col in df.columns:
    if col not in ["id", "stroke"]: 
        unique_vals = sorted(df[col].dropna().unique().tolist(), key=lambda x: str(x))
        category_dict[col] = unique_vals

# 保存为 JSON
with open(json_file, "w", encoding="utf-8") as f:
    json.dump(category_dict, f, indent=2, ensure_ascii=False)

print(f" {json_file}")
