# -*- coding: utf-8 -*-
"""
QTNM Prototype (Quantum Trade Negotiation Model)
Single-file script that generates:
  - synthetic_coefficients.json (synthetic response model)
  - meta.json (model meta)
  - pareto_frontier_decisions.csv (frontier decisions)
  - negotiation_menus.csv (three representative menus)
  - qnfi_timeseries.csv (weekly fairness index)
  - pareto_frontier.png (frontier chart)
  - menus_on_frontier.png (menus overlay chart)
  - qnfi_weekly.png (Q-NFI timeseries chart)
  - dashboard.html (static demo dashboard)
  - Q-NFI_weekly_report_template.md (markdown report boilerplate)

Usage:
    python qtnm_prototype.py
Outputs are written to: ./output
"""
import os, json, math, random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------
# Configuration
# --------------------------
def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)
    return p

ROOT = os.path.dirname(__file__)
OUT = ensure_dir(os.path.join(ROOT, "output"))
random.seed(2025); np.random.seed(2025)

META = {
    "n_tau": 10,     # tariff sectors (binary: 1 = reduce tariff in sector i)
    "n_ntb": 8,      # non-tariff barriers (binary: 1 = relax barrier j)
    "n_tech": 6,     # tech/FDI restrictions (binary: 1 = ease restriction k)
    "n_access": 6,   # market access/transparency (binary: 1 = commit l)
    "delta_t_options": [30, 60, 90]  # phase spacing in days
}

# --------------------------
# Synthetic response model
# --------------------------
# Signs are chosen to reflect intuition in a simplified manner.
COEF = {
    "us_cpi":      list(np.random.uniform(-1.2, -0.2, size=META["n_tau"])),   # tariff cut -> lower CPI (better for US price stability)
    "us_jobs":     list(np.random.uniform( 0.1,  1.0, size=META["n_tau"])),   # tariff cut -> helps jobs
    "us_jobs_acc": list(np.random.uniform( 0.2,  1.2, size=META["n_access"])),# access commitments -> help jobs/investment
    "us_sec":      list(np.random.uniform(-1.0, -0.1, size=META["n_tech"])),  # easing tech restrictions -> raises dependency (bad), negative sign
    "cn_export":   list(np.random.uniform( 0.3,  1.3, size=META["n_tau"])),   # tariff cut -> boosts CN exports
    "cn_tech":     list(np.random.uniform( 0.4,  1.4, size=META["n_tech"])),  # easing tech/FDI -> improves CN tech access
    "cn_supply":   list(np.random.uniform( 0.2,  1.0, size=META["n_ntb"])),   # lowering NTBs -> stabilizes CN supply
    "ntb_cost":    list(np.random.uniform(-0.8, -0.1, size=META["n_ntb"]))    # NTB relax may also ease some US frictions (small negative to CPI)
}

with open(os.path.join(OUT, "meta.json"), "w", encoding="utf-8") as f:
    json.dump(META, f, indent=2, ensure_ascii=False)
with open(os.path.join(OUT, "synthetic_coefficients.json"), "w", encoding="utf-8") as f:
    json.dump(COEF, f, indent=2, ensure_ascii=False)

# --------------------------
# Decision generation & feasibility
# --------------------------
def sample_decision(meta):
    d = {
        "tau":    np.random.randint(0, 2, size=meta["n_tau"]),     # 1 = reduce tariff
        "ntb":    np.random.randint(0, 2, size=meta["n_ntb"]),     # 1 = relax NTB
        "tech":   np.random.randint(0, 2, size=meta["n_tech"]),    # 1 = ease tech restriction
        "access": np.random.randint(0, 2, size=meta["n_access"]),  # 1 = commit access/transparency
        "delta_t": int(np.random.choice(meta["delta_t_options"]))  # 30/60/90
    }
    return d

def check_constraints(d):
    # Example political/verification feasibility constraints for a 90-day window
    # 1) Tech easing at most 3 at a time
    if d["tech"].sum() > 3:
        return False
    # 2) Access commitments proportional to tariff cuts (simple band)
    if d["access"].sum() < max(1, int(0.4 * d["tau"].sum())):
        return False
    # 3) If too many simultaneous changes, avoid the shortest pace (30 days)
    if d["tau"].sum() + d["ntb"].sum() > 12 and d["delta_t"] == 30:
        return False
    # 4) At least some action in any package
    if d["tau"].sum() + d["ntb"].sum() + d["tech"].sum() + d["access"].sum() == 0:
        return False
    return True

# --------------------------
# Objective evaluation (US, CN)
# --------------------------
COEF_NP = {k: np.array(v) for k, v in COEF.items()}

def squash_positive(x, scale):
    # Soft saturation into [0,1]. Larger x -> closer to 1.
    return 1.0 - math.exp(-max(0.0, x) / (scale + 1e-9))

def evaluate_objectives(d):
    # US side (maximize): price stability, jobs/investment, security
    us_price   = - (d["tau"] @ COEF_NP["us_cpi"] + d["ntb"] @ COEF_NP["ntb_cost"]) # lower CPI => higher score via minus sign
    us_jobs    =   (d["tau"] @ COEF_NP["us_jobs"]) + (d["access"] @ COEF_NP["us_jobs_acc"])
    us_sec_pen =    d["tech"] @ (-COEF_NP["us_sec"])  # negative coef -> easing adds dependency; penalize proportionally
    us_sec     = max(0.0, 1.0 - (us_sec_pen / (len(COEF_NP["us_sec"]) * np.abs(COEF_NP["us_sec"]).mean() + 1e-9)))
    us_score   = 0.4 * squash_positive(us_price, 8.0) + 0.4 * squash_positive(us_jobs, 10.0) + 0.2 * us_sec

    # CN side (maximize): exports, tech access, supply stability
    cn_exp   = d["tau"]  @ COEF_NP["cn_export"]
    cn_tech  = d["tech"] @ COEF_NP["cn_tech"]
    cn_sup   = d["ntb"]  @ COEF_NP["cn_supply"]
    cn_score = 0.4 * squash_positive(cn_exp, 10.0) + 0.35 * squash_positive(cn_tech, 8.0) + 0.25 * squash_positive(cn_sup, 6.0)

    return float(us_score), float(cn_score)

# --------------------------
# Pareto tools
# --------------------------
def dominates(a, b):
    # a dominates b if a>=b elementwise and strictly better in at least one dim
    return (a[0] >= b[0] and a[1] >= b[1]) and (a[0] > b[0] or a[1] > b[1])

def pareto_front(points):
    # points: list of dict {'us':..,'cn':..,'decision':..}
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
# Search: global sampling + local refinement
# --------------------------
def local_refine(sol, meta, iters=200):
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

# Global sampling (emulate annealing sweep)
candidates = []
for _ in range(2000):
    d0 = sample_decision(META)
    if not check_constraints(d0):
        continue
    us0, cn0 = evaluate_objectives(d0)
    candidates.append({"us": us0, "cn": cn0, "decision": d0})

front0 = pareto_front(candidates)
refined = [local_refine(s, META, iters=250) for s in front0[:100]]  # refine top 100 seeds
front = pareto_front(refined + front0)

# --------------------------
# Save frontier decisions and menus
# --------------------------
def decisions_to_df(points):
    rows = []
    for i, p in enumerate(points):
        d = p["decision"]
        rows.append({
            "idx": i, "us_score": p["us"], "cn_score": p["cn"], "delta_t": d["delta_t"],
            **{f"tau_{k}": int(v) for k, v in enumerate(d["tau"]) },
            **{f"ntb_{k}": int(v) for k, v in enumerate(d["ntb"]) },
            **{f"tech_{k}": int(v) for k, v in enumerate(d["tech"]) },
            **{f"access_{k}": int(v) for k, v in enumerate(d["access"]) },
        })
    return pd.DataFrame(rows)

df_front = decisions_to_df(front)
df_front.to_csv(os.path.join(OUT, "pareto_frontier_decisions.csv"), index=False)

def pick_menus(front):
    if len(front) == 0:
        return []
    if len(front) <= 3:
        return front
    return [front[0], front[len(front)//2], front[-1]]  # Conservative / Balanced / Ambitious

menus = pick_menus(front)
df_menus = decisions_to_df(menus)
df_menus.to_csv(os.path.join(OUT, "negotiation_menus.csv"), index=False)

# --------------------------
# Charts (Matplotlib) — one chart per figure; no explicit colors
# --------------------------
plt.figure(figsize=(7,5))
plt.scatter([p["us"] for p in candidates], [p["cn"] for p in candidates], s=10, alpha=0.3)
plt.scatter([p["us"] for p in front], [p["cn"] for p in front], s=30)
plt.xlabel("US Composite Score"); plt.ylabel("CN Composite Score"); plt.title("Pareto Frontier (US vs CN Composite Scores)")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "pareto_frontier.png"), dpi=160); plt.close()

plt.figure(figsize=(7,5))
plt.scatter([p["us"] for p in front], [p["cn"] for p in front], s=18, alpha=0.7)
for m, label in zip(menus, ["Conservative","Balanced","Ambitious"][:len(menus)]):
    plt.scatter([m["us"]], [m["cn"]], s=90)
    plt.text(m["us"], m["cn"], label)
plt.xlabel("US Composite Score"); plt.ylabel("CN Composite Score"); plt.title("Negotiation Menus on Frontier")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "menus_on_frontier.png"), dpi=160); plt.close()

# --------------------------
# Q-NFI (max-min fairness over frontier) timeseries
# --------------------------
def qnfi(front):
    return max(min(p["us"], p["cn"]) for p in front) if front else 0.0

weeks = [datetime.today() - timedelta(days=7*i) for i in range(12, -1, -1)]
weeks_str = [w.strftime("%Y-%m-%d") for w in weeks]
cur_qnfi = qnfi(front)
# Simulate a plausible trajectory around current value
hist = [max(0.0, min(1.0, cur_qnfi + np.random.normal(0.0, 0.05) + 0.02*(i-6))) for i in range(len(weeks))]
pd.DataFrame({"week": weeks_str, "QNFI": hist}).to_csv(os.path.join(OUT, "qnfi_timeseries.csv"), index=False)

plt.figure(figsize=(7,4))
plt.plot(weeks_str, hist, marker="o")
plt.xticks(rotation=45); plt.ylabel("Q-NFI"); plt.title("Quantum Negotiation Frontier Index (Weekly)")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "qnfi_weekly.png"), dpi=160); plt.close()

# --------------------------
# Static HTML dashboard
# --------------------------
menus_table_html = df_menus.to_html(index=False)
front_head_html = df_front[["idx","us_score","cn_score","delta_t"]].head(15).to_html(index=False)

html = (
    "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
    "<title>QTNM Dashboard</title>"
    "<meta name='viewport' content='width=device-width, initial-scale=1.0'>"
    "<style>"
    "body{font-family:-apple-system,Arial,Helvetica,sans-serif;margin:24px;}"
    "h1,h2{margin-top:20px;} img{max-width:100%;height:auto;} "
    "table{border-collapse:collapse;width:100%;} th,td{border:1px solid #ccc;padding:6px 8px;text-align:center;}"
    ".small{color:#555;font-size:0.9em;}"
    "</style></head><body>"
    "<h1>Quantum Trade Negotiation Model (QTNM) — Dashboard</h1>"
    f"<p class='small'>Build time: {datetime.utcnow().isoformat()}Z</p>"
    "<h2>Pareto Frontier</h2><p>US Composite Score vs CN Composite Score</p>"
    "<img src='pareto_frontier.png' alt='Pareto Frontier'>"
    "<h2>Negotiation Menus</h2><p>Three representative frontier packages</p>"
    "<img src='menus_on_frontier.png' alt='Menus on Frontier'>"
    f"{menus_table_html}"
    "<h2>Q-NFI (Quantum Negotiation Frontier Index)</h2>"
    "<img src='qnfi_weekly.png' alt='Q-NFI Weekly'>"
    "<p class='small'>Q-NFI = max<sub>x ∈ Frontier</sub> min(US(x), CN(x))</p>"
    "<h2>Frontier Head (Top 15)</h2>"
    f"{front_head_html}"
    "<hr><p class='small'>DISCLAIMER: Synthetic coefficients for demo. Replace with empirical elasticities when deploying.</p>"
    "</body></html>"
)
with open(os.path.join(OUT, "dashboard.html"), "w", encoding="utf-8") as f:
    f.write(html)

# --------------------------
# Q-NFI weekly report template (Markdown)
# --------------------------
md = f"""# 量子谈判前沿指数（Q-NFI）周报模板

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
- 我们将谈判条款组合映射为多层次 Ising 哈密顿量，使用“退火式全局搜索 + 变分式精修”探索帕累托前沿；
- 共同约束（合规、政治、可验证性）通过惩罚项并入能量函数；
- Q-NFI 定义为前沿上“美中得分的最小值”的最大化，用于衡量“可被双方同时接受”的公平度。

## 五、下周展望
- 重点行业/清单调整：{{NEXT_WEEK_FOCUS}}
- 数据缺口与采集计划：{{DATA_NEEDS}}
- 预期会谈节奏与触发器：{{MEETING_CADENCE}}
"""
with open(os.path.join(OUT, "Q-NFI_weekly_report_template.md"), "w", encoding="utf-8") as f:
    f.write(md)

print("Done. Outputs written to:", OUT)
