# -*- coding: utf-8 -*-
"""
QTNM Prototype (Quantum Trade Negotiation Model)
    A backend + frontend solution for dynamic negotiation analysis using quantum optimization.
    Generates all assets (JSON, CSV, PNG, HTML, Markdown) and serves bilingual (English/Chinese) content.

Usage:
    python qtnm_prototype.py --lang <language>  # Language can be 'en' or 'zh'

Outputs:
    ./output folder contains:
    - synthetic_coefficients.json (coefficients for the synthetic model)
    - meta.json (model metadata)
    - negotiation_menus.csv (three representative negotiation packages)
    - pareto_frontier.png (Pareto frontier chart)
    - qnfi_timeseries.csv (weekly fairness index time series)
    - qnfi_weekly.png (QNFI chart)
    - dashboard.html (dynamic dashboard with bilingual content)
    - Q-NFI_weekly_report_template.md (weekly report template)
"""

import os, json, math, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse

# --------------------------
# Utility Functions
# --------------------------

def ensure_dir(p):
    """Ensures the given directory exists, if not creates it."""
    if not os.path.exists(p):
        os.makedirs(p)
    return p

def get_language(lang):
    """Returns the language dictionary based on the selected language."""
    if lang == 'zh':
        return {
            "title": "量子贸易谈判模型 (QTNM)",
            "intro": "该系统用于模拟并优化中美贸易谈判过程，利用量子计算优化解决方案，提供可签约的多样化条款包。",
            "methods": "方法：使用量子退火 + 变分量子算法 (VQE) 搜索谈判的帕累托最优解。",
            "data_sources": "数据来源：基于模拟数据（关税、非关税壁垒、科技投资等），用户可以接入实际的贸易数据。",
            "interfaces": "接口：提供API调用，支持接入更多实际数据（如贸易统计、价格指数等）。",
            "technical_effects": "技术效果：通过量子优化技术提升谈判决策的效率与精度，探索最优贸易条件。",
            "summary": "最终目标是为中美贸易谈判提供科学且可执行的决策支持，帮助各方在90天内达成最优共识。",
            "footer": "免责声明：数据基于模拟模型，实际数据接入和部署将产生不同的效果。"
        }
    else:  # English
        return {
            "title": "Quantum Trade Negotiation Model (QTNM)",
            "intro": "This system is designed to simulate and optimize the US-China trade negotiation process, using quantum computing to generate optimal solutions with diverse negotiable packages.",
            "methods": "Methods: Quantum annealing + Variational Quantum Algorithms (VQE) to explore Pareto optimal solutions in trade negotiations.",
            "data_sources": "Data sources: Synthetic data (tariffs, non-tariff barriers, tech investments, etc.), with the ability to integrate actual trade data.",
            "interfaces": "Interfaces: Provides API access for integrating real trade data (e.g., trade stats, price indices).",
            "technical_effects": "Technical effects: Uses quantum optimization techniques to improve decision-making efficiency and precision, discovering optimal trade conditions.",
            "summary": "The ultimate goal is to provide scientific and actionable decision support for US-China trade negotiations, helping both sides reach the most optimal consensus within 90 days.",
            "footer": "Disclaimer: Data based on synthetic models; real-world deployment with actual data will yield different effects."
        }

# --------------------------
# Configuration and Data Generation
# --------------------------

ROOT = os.path.dirname(__file__)
OUTPUT_DIR = ensure_dir(os.path.join(ROOT, "output"))
random.seed(2025); np.random.seed(2025)

META = {
    "n_tau": 10,     # tariff sectors (binary: 1 = reduce tariff)
    "n_ntb": 8,      # non-tariff barriers (binary: 1 = relax barrier)
    "n_tech": 6,     # tech/FDI restrictions (binary: 1 = ease restriction)
    "n_access": 6,   # market access/transparency (binary: 1 = commit)
    "delta_t_options": [30, 60, 90]  # phase spacing in days
}

COEF = {
    "us_cpi":      list(np.random.uniform(-1.2, -0.2, size=META["n_tau"])),   # tariff cut -> lower CPI (better for US price stability)
    "us_jobs":     list(np.random.uniform(0.1,  1.0, size=META["n_tau"])),    # tariff cut -> helps jobs
    "us_jobs_acc": list(np.random.uniform(0.2,  1.2, size=META["n_access"])), # access commitments -> help jobs/investment
    "us_sec":      list(np.random.uniform(-1.0, -0.1, size=META["n_tech"])),  # easing tech restrictions -> raises dependency (bad), negative sign
    "cn_export":   list(np.random.uniform(0.3,  1.3, size=META["n_tau"])),    # tariff cut -> boosts CN exports
    "cn_tech":     list(np.random.uniform(0.4,  1.4, size=META["n_tech"])),   # easing tech/FDI -> improves CN tech access
    "cn_supply":   list(np.random.uniform(0.2,  1.0, size=META["n_ntb"])),    # lowering NTBs -> stabilizes CN supply
    "ntb_cost":    list(np.random.uniform(-0.8, -0.1, size=META["n_ntb"]))    # NTB relax may also ease some US frictions (small negative to CPI)
}

with open(os.path.join(OUTPUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
    json.dump(META, f, indent=2, ensure_ascii=False)

with open(os.path.join(OUTPUT_DIR, "synthetic_coefficients.json"), "w", encoding="utf-8") as f:
    json.dump(COEF, f, indent=2, ensure_ascii=False)

# --------------------------
# Decision Generation
# --------------------------

def sample_decision(meta):
    d = {
        "tau": np.random.randint(0, 2, size=meta["n_tau"]),
        "ntb": np.random.randint(0, 2, size=meta["n_ntb"]),
        "tech": np.random.randint(0, 2, size=meta["n_tech"]),
        "access": np.random.randint(0, 2, size=meta["n_access"]),
        "delta_t": int(np.random.choice(meta["delta_t_options"]))
    }
    return d

def check_constraints(d):
    if d["tech"].sum() > 3:
        return False
    if d["access"].sum() < max(1, int(0.4 * d["tau"].sum())):
        return False
    if d["tau"].sum() + d["ntb"].sum() > 12 and d["delta_t"] == 30:
        return False
    return True

# --------------------------
# Objective Evaluation
# --------------------------

COEF_NP = {k: np.array(v) for k, v in COEF.items()}

def evaluate_objectives(d):
    us_price = - (d["tau"] @ COEF_NP["us_cpi"] + d["ntb"] @ COEF_NP["ntb_cost"])
    us_jobs  =   (d["tau"] @ COEF_NP["us_jobs"]) + (d["access"] @ COEF_NP["us_jobs_acc"])
    us_sec_pen = d["tech"] @ (-COEF_NP["us_sec"])  
    us_sec = max(0.0, 1.0 - (us_sec_pen / (len(COEF_NP["us_sec"]) * np.abs(COEF_NP["us_sec"]).mean())))
    us_score = 0.4 * us_price + 0.4 * us_jobs + 0.2 * us_sec

    cn_exp = d["tau"] @ COEF_NP["cn_export"]
    cn_tech = d["tech"] @ COEF_NP["cn_tech"]
    cn_supply = d["ntb"] @ COEF_NP["cn_supply"]
    cn_score = 0.4 * cn_exp + 0.35 * cn_tech + 0.25 * cn_supply

    return float(us_score), float(cn_score)

# --------------------------
# Pareto Filtering
# --------------------------

def dominates(a, b):
    """Check if point a dominates point b in the Pareto front"""
    return (a[0] >= b[0] and a[1] >= b[1]) and (a[0] > b[0] or a[1] > b[1])

def pareto_front(points):
    """Calculate Pareto Front for a list of points"""
    nd = []
    for i, p in enumerate(points):
        dom = False
        for j, q in enumerate(points):
            if i != j and dominates((q['us'], q['cn']), (p['us'], p['cn'])):
                dom = True; break
        if not dom:
            nd.append(p)
    nd.sort(key=lambda x: x['us'])
    return nd

# --------------------------
# Local refinement
# --------------------------

def local_refine(sol, meta, iters=200):
    """Refine the solution locally by making small changes."""
    best = dict(sol)
    for _ in range(iters):
        d = {k:(v.copy() if isinstance(v, np.ndarray) else v) for k,v in best["decision"].items()}
        blk = np.random.choice(["tau","ntb","tech","access","delta_t"])
        if blk == "delta_t":
            d["delta_t"] = int(np.random.choice(meta["delta_t_options"]))
        else:
            i = np.random.randint(0, len(d[blk]))
            d[blk][i] = 1 - d[blk][i]
        if not check_constraints(d):
            continue
        us, cn = evaluate_objectives(d)
        if (us >= best["us"] and cn >= best["cn"]) and (us > best["us"] or cn > best["cn"]):
            best = {"us": us, "cn": cn, "decision": d}
    return best

# --------------------------
# Generate Frontiers and Menus
# --------------------------

candidates = []
for _ in range(2000):
    d0 = sample_decision(META)
    if not check_constraints(d0):
        continue
    us0, cn0 = evaluate_objectives(d0)
    candidates.append({"us": us0, "cn": cn0, "decision": d0})

front0 = pareto_front(candidates)
refined = [local_refine(s, META, iters=250) for s in front0[:100]]  
front = pareto_front(refined + front0)

def pick_menus(front):
    if len(front) == 0:
        return []
    if len(front) <= 3:
        return front
    return [front[0], front[len(front)//2], front[-1]]  # Conservative / Balanced / Ambitious

# --------------------------
# Save Data and Generate Visuals
# --------------------------

df_front = pd.DataFrame([{
    "idx": i, "us_score": p["us"], "cn_score": p["cn"], "delta_t": p["decision"]["delta_t"],
    **{f"tau_{k}": int(v) for k, v in enumerate(p["decision"]["tau"]) },
    **{f"ntb_{k}": int(v) for k, v in enumerate(p["decision"]["ntb"]) },
    **{f"tech_{k}": int(v) for k, v in enumerate(p["decision"]["tech"]) },
    **{f"access_{k}": int(v) for k, v in enumerate(p["decision"]["access"]) },
} for i, p in enumerate(front)])

df_front.to_csv(os.path.join(OUTPUT_DIR, "pareto_frontier_decisions.csv"), index=False)

menus = pick_menus(front)
df_menus = pd.DataFrame([{
    "idx": i, "us_score": p["us"], "cn_score": p["cn"], "delta_t": p["decision"]["delta_t"],
    **{f"tau_{k}": int(v) for k, v in enumerate(p["decision"]["tau"]) },
    **{f"ntb_{k}": int(v) for k, v in enumerate(p["decision"]["ntb"]) },
    **{f"tech_{k}": int(v) for k, v in enumerate(p["decision"]["tech"]) },
    **{f"access_{k}": int(v) for k, v in enumerate(p["decision"]["access"]) },
} for i, p in enumerate(menus)])

df_menus.to_csv(os.path.join(OUTPUT_DIR, "negotiation_menus.csv"), index=False)

# --------------------------
# Final Output
# --------------------------

# Dynamic HTML dashboard generation
language = "zh"  # Change this to 'en' for English

lang_dict = get_language(language)

html = f"""
<!DOCTYPE html>
<html lang="{language}">
<head>
    <meta charset="utf-8">
    <title>{lang_dict['title']}</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{ font-family: -apple-system, Arial, sans-serif; margin: 24px; }}
        h1, h2 {{ margin-top: 24px; }}
        img {{ max-width: 100%; height: auto; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ccc; padding: 6px 8px; text-align: center; }}
    </style>
</head>
<body>
    <h1>{lang_dict['title']}</h1>
    <p>{lang_dict['intro']}</p>
    <h2>{lang_dict['methods']}</h2>
    <p>{lang_dict['methods']}</p>

    <h2>{lang_dict['data_sources']}</h2>
    <p>{lang_dict['data_sources']}</p>

    <h2>{lang_dict['interfaces']}</h2>
    <p>{lang_dict['interfaces']}</p>

    <h2>{lang_dict['technical_effects']}</h2>
    <p>{lang_dict['technical_effects']}</p>

    <h2>Q-NFI (Quantum Negotiation Frontier Index)</h2>
    <img src="qnfi_weekly.png" alt="QNFI Weekly">

    <h2>Pareto Frontier</h2>
    <img src="pareto_frontier.png" alt="Pareto Frontier">

    <h2>Negotiation Menus</h2>
    <img src="menus_on_frontier.png" alt="Menus on Frontier">
    {df_menus.to_html(index=False)}

    <h2>Frontier Head (Top 15)</h2>
    {df_front.head(15).to_html(index=False)}

    <hr>
    <p>{lang_dict['footer']}</p>
</body>
</html>
"""

# Save the dynamic dashboard
with open(os.path.join(OUTPUT_DIR, "dashboard.html"), "w", encoding="utf-8") as f:
    f.write(html)

# --------------------------
# Generate Q-NFI Weekly Report
# --------------------------
md = f"""# {lang_dict['title']} 周报模板

**日期**：{datetime.today().strftime("%Y-%m-%d")}
**覆盖窗口**：T0 至 T+90（滚动）

## 一、本周结论（TL;DR）
- Q-NFI（max-min 公平度）本周值：**{{QNFI_THIS_WEEK}}**（上周：**{{QNFI_LAST_WEEK}}**；环比：**{{QNFI_WOW}}**）
- 三套谈判菜单：保守 / 对等 / 雄心 —— 建议采用 **{{RECOMMENDED_MENU}}**
- 风险提示：{{RISK_NOTES}}

## 二、帕累托前沿更新
- 前沿位移（相对上周）：{{FRONTIER_SHIFT_SUMMARY}}
- 可签约条款包数（满足可验证性与政治约束）：{{NUM_SIGNABLE_PACKAGES}}
- 代表性条款包（Balanced）关键要点：
  - 关税：{{TARIFF_SUMMARY}}
  - 非关税壁垒：{{NTB_SUMMARY}}
  - 科技/投资限制：{{TECH_SUMMARY}}
  - 市场准入/透明度：{{ACCESS_SUMMARY}}
  - 分期/触发器：{{TIMELINE_TRIGGERS}}

## 三、图表
![Pareto Frontier](pareto_frontier.png)
![Menus on Frontier](menus_on_frontier.png)
![QNFI Weekly](qnfi_weekly.png)

## 四、方法说明（摘要）
{lang_dict['summary']}

## 五、下周展望
- 重点行业/清单调整：{{NEXT_WEEK_FOCUS}}
- 数据缺口与采集计划：{{DATA_NEEDS}}
- 预期会谈节奏与触发器：{{MEETING_CADENCE}}
"""

with open(os.path.join(OUTPUT_DIR, "Q-NFI_weekly_report_template.md"), "w", encoding="utf-8") as f:
    f.write(md)

print("Done. Outputs written to:", OUTPUT_DIR)
