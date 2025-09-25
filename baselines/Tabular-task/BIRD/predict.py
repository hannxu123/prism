import json
import pandas as pd

with open('code/bird/data_bird/cpt_trained_adult_gemini.json', 'r', encoding='utf-8') as f:
    cpt = json.load(f)

with open("code/bird/data_bird/factors_adult.json", "r", encoding="utf-8") as f:
    factors = json.load(f)

test_samples = pd.read_csv("code/bird/data_bird/adult_300_binned.csv")

results = []
for _, row in test_samples.iterrows():
    fvals = {factor: row[factor] for factor in factors if factor in row}
    
    prod_app = prod_rej = 1.0
    for factor, val in fvals.items():
        info = cpt.get(factor, {}).get(str(val), {"income_over_50": 0.5, "income_lower_50": 0.5})
        p_app = info.get("income_over_50", 0.5)
        p_rej = info.get("income_lower_50", 0.5)

        prod_app *= p_app
        prod_rej *= p_rej

    # Normalize
    total = prod_app + prod_rej
    if total > 0:
        p_app = prod_app / total
        p_rej = prod_rej / total
    else:
        p_app, p_rej = 0.5, 0.5

    pred = 1 if p_app >= p_rej else 0
    # results.append({
    #     "extracted_factors": fvals,
    #     "p_income_over_50": round(p_app, 3),
    #     "p_income_lower_50": round(p_rej, 3),
    #     "prediction": pred,
    #     "label": row.get("stroke", ""),
    #     "correct": pred == row.get("stroke", "")
    # })
    results.append({
        "extracted_factors": fvals,
        "prediction": round(p_app, 3),
        # "label": row.get("stroke", "")
        "label": int(row.get("income", "") == '>50K'),
    })

with open("code/bird/data_bird/prediction_adult_trained_gemini2.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
