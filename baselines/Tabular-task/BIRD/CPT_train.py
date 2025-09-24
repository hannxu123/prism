import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import pipeline
import re
import math


with open('code/bird/data_bird/cpt_init_adult_gemini.json', 'r', encoding='utf-8') as f:
    cpt_init = json.load(f)

with open("code/bird/data_bird/factors_adult.json", "r", encoding="utf-8") as f:
    factors = json.load(f)

with open("code/bird/data_bird/synthetic_train_adult_gemini.json", "r", encoding="utf-8") as f:
    samples = json.load(f)

# for s in samples:
#     label = s.get("label", "").lower()
#     s["y"] = 1.0 if label == "income_over_50" else 0.0

for s in samples:
    label = s.get("prob", "")
    s["y"] = label

factor_names = list(factors.keys())
param_indices = {}  # (factor, value) -> idx
idx = 0
for f in factor_names:
    for v in factors[f]:
        param_indices[(f, v)] = idx
        idx += 1
n_params = idx

# Initialize theta from CPT_init by logit transform
theta = torch.zeros(n_params, requires_grad=True)
for (f, v), i in param_indices.items():
    p = cpt_init[f][v]["income_over_50"] ##################################
    # clamp to (ε,1-ε)
    eps = 1e-3
    p_clamped = min(max(p, eps), 1-eps)
    theta.data[i] = math.log(p_clamped/(1-p_clamped))

# 4. Training setup
optimizer = optim.SGD([theta], lr=1e-2)
alpha = 1  # balance coefficient
eps_mr = 0
epochs = 200
batch_size = 4

# def compute_P_est(fvals):
#     idxs = [param_indices[(f, fvals[f])] for f in factor_names]
#     ps = torch.sigmoid(theta[idxs])
#     prod_p = torch.prod(ps)
#     prod_1p = torch.prod(1-ps)
#     return prod_p / (prod_p + prod_1p)

def compute_P_est(fvals):
    ps = []
    for f in factor_names:
        val = fvals.get(f, None)
        key = (f, val)
        if key in param_indices:
            # 如果匹配上，就用训练过的参数
            idx = param_indices[key]
            p_j = torch.sigmoid(theta[idx])
        else:
            # 匹配不上就用 0.5
            p_j = torch.tensor(0.5, device=theta.device)
        ps.append(p_j)
    # 把所有单因子概率乘起来
    ps = torch.stack(ps)
    prod_p  = torch.prod(ps)
    prod_1p = torch.prod(1 - ps)
    return prod_p / (prod_p + prod_1p)


# def compute_P_est(fvals):
#     """
#     计算 P(O|f) 时，如果某因子或取值找不到，则使用 p_j=0.5
#     """
#     p_list = []
#     for f in factor_names:
#         val = fvals.get(f, None)
#         key = (f, val)
#         if val is None or key not in param_indices:
#             # 缺失或者 key 不存在，默认 p=0.5
#             p_j = torch.tensor(0.5, device=theta.device)
#         else:
#             idx = param_indices[key]
#             p_j = torch.sigmoid(theta[idx])
#         p_list.append(p_j)
#     ps = torch.stack(p_list)
#     prod_p = torch.prod(ps)
#     prod_not_p = torch.prod(1 - ps)
#     return prod_p / (prod_p + prod_not_p)

# def compute_P_trained():
#     sum_est = torch.zeros(n_params)
#     count = torch.zeros(n_params)
#     for s in samples:
#         P = compute_P_est(s["fvals"]).detach()
#         for f, v in s["fvals"].items():
#             i = param_indices[(f, v)]
#             sum_est[i] += P
#             count[i] += 1
#     return sum_est / torch.where(count>0, count, torch.ones_like(count))
def compute_P_trained():
    sum_est = torch.zeros(n_params, device=theta.device)
    count   = torch.zeros(n_params, device=theta.device)
    for s in samples:
        P = compute_P_est(s["fvals"]).detach()
        for f, v in s["fvals"].items():
            key = (f, v)
            if key not in param_indices:
                # 跳过所有在 param_indices 中不存在的 (factor, value)
                continue
            i = param_indices[key]
            sum_est[i] += P
            count[i]   += 1
    # 对于 count=0 的参数，保持 sum/count = 0
    return sum_est / torch.where(count > 0, count, torch.ones_like(count))

# --- 5. Train ---
for epoch in range(1, epochs+1):
    random.shuffle(samples)
    total_loss = 0.0
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        optimizer.zero_grad()

        # MSE
        mse = 0.0
        for s in batch:
            P_est = compute_P_est(s["fvals"])
            mse += (P_est - s["y"])**2
        mse = mse / len(batch)

        # Margin ranking
        P_tr = compute_P_trained()
        ytarget = torch.tensor([
            +1.0 if cpt_init[f][v]["income_over_50"]>0.5 else -1.0
            for (f,v),_ in sorted(param_indices.items(), key=lambda x:x[1])
        ])
        mr = torch.clamp(-ytarget * (torch.sigmoid(theta)-0.5) + eps_mr, min=0.0).mean()

        loss = mse + alpha*mr
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch:02d}, avg loss: {total_loss/(len(samples)//batch_size):.4f}")

# --- 6. Save trained CPT ---
trained = {}
for (f, v), i in param_indices.items():
    p = float(torch.sigmoid(theta[i]).item())
    trained.setdefault(f, {})[v] = {"income_over_50": p, "income_lower_50": 1-p}
with open("code/bird/data_bird/cpt_trained_adult_gemini.json", "w", encoding="utf-8") as f:
    json.dump(trained, f, ensure_ascii=False, indent=2)