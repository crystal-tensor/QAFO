import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import random
from matplotlib.animation import FuncAnimation
from IPython.display import Image

# Step 1: Set up the configuration
meta = {
    "n_tau": 10,     # tariff sectors (binary: 1 = reduce tariff in sector i)
    "n_ntb": 8,      # non-tariff barriers (binary: 1 = relax barrier j)
    "n_tech": 6,     # tech/FDI restrictions (binary: 1 = ease restriction k)
    "n_access": 6,   # market access/transparency (binary: 1 = commit l)
    "delta_t_options": [30, 60, 90]  # phase spacing in days
}

# Step 2: Define the synthetic response model (coefficients)
COEF = {
    "us_cpi": list(np.random.uniform(-1.2, -0.2, size=meta["n_tau"])),
    "us_jobs": list(np.random.uniform(0.1, 1.0, size=meta["n_tau"])),
    "us_jobs_acc": list(np.random.uniform(0.2, 1.2, size=meta["n_access"])),
    "us_sec": list(np.random.uniform(-1.0, -0.1, size=meta["n_tech"])),
    "cn_export": list(np.random.uniform(0.3, 1.3, size=meta["n_tau"])),
    "cn_tech": list(np.random.uniform(0.4, 1.4, size=meta["n_tech"])),
    "cn_supply": list(np.random.uniform(0.2, 1.0, size=meta["n_ntb"])),
    "ntb_cost": list(np.random.uniform(-0.8, -0.1, size=meta["n_ntb"]))
}

# Save the coefficients to a JSON file
os.makedirs("output", exist_ok=True)
with open("output/synthetic_coefficients.json", "w", encoding="utf-8") as f:
    json.dump(COEF, f, indent=2, ensure_ascii=False)

# Step 3: Define the decision-making process
def sample_decision(meta):
    return {
        "tau": np.random.randint(0, 2, size=meta["n_tau"]),
        "ntb": np.random.randint(0, 2, size=meta["n_ntb"]),
        "tech": np.random.randint(0, 2, size=meta["n_tech"]),
        "access": np.random.randint(0, 2, size=meta["n_access"]),
        "delta_t": int(np.random.choice(meta["delta_t_options"]))
    }

# Step 4: Define the objective evaluation function for US and China
COEF_NP = {k: np.array(v) for k, v in COEF.items()}

def squash_positive(x, scale):
    return 1.0 - math.exp(-max(0.0, x) / (scale + 1e-9))

def evaluate_objectives(d):
    us_price = -(d["tau"] @ COEF_NP["us_cpi"] + d["ntb"] @ COEF_NP["ntb_cost"])
    us_jobs = (d["tau"] @ COEF_NP["us_jobs"]) + (d["access"] @ COEF_NP["us_jobs_acc"])
    us_sec_pen = d["tech"] @ (-COEF_NP["us_sec"])
    us_sec = max(0.0, 1.0 - (us_sec_pen / (len(COEF_NP["us_sec"]) * np.abs(COEF_NP["us_sec"]).mean() + 1e-9)))
    us_score = 0.4 * squash_positive(us_price, 8.0) + 0.4 * squash_positive(us_jobs, 10.0) + 0.2 * us_sec

    cn_exp = d["tau"] @ COEF_NP["cn_export"]
    cn_tech = d["tech"] @ COEF_NP["cn_tech"]
    cn_sup = d["ntb"] @ COEF_NP["cn_supply"]
    cn_score = 0.4 * squash_positive(cn_exp, 10.0) + 0.35 * squash_positive(cn_tech, 8.0) + 0.25 * squash_positive(cn_sup, 6.0)

    return float(us_score), float(cn_score)

# Step 5: Simulate trade negotiation and find the Pareto front
def dominates(a, b):
    """Check if solution a dominates solution b"""
    return all(a[i] >= b[i] for i in range(len(a))) and any(a[i] > b[i] for i in range(len(a)))

def pareto_front(points):
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

# Step 6: Run the simulation to generate possible negotiation strategies and evaluate them
candidates = []
for _ in range(2000):
    d = sample_decision(meta)
    us, cn = evaluate_objectives(d)
    candidates.append({"us": us, "cn": cn, "decision": d})

front0 = pareto_front(candidates)

# Step 7: Define the visualization of the negotiation process
fig, ax = plt.subplots(figsize=(7, 5))

def update(frame):
    ax.clear()
    ax.scatter([p["us"] for p in candidates], [p["cn"] for p in candidates], s=10, alpha=0.3)
    ax.scatter([p["us"] for p in front0], [p["cn"] for p in front0], s=30)
    ax.set_xlabel("US Composite Score")
    ax.set_ylabel("CN Composite Score")
    ax.set_title("Pareto Frontier (US vs CN Composite Scores)")

ani = FuncAnimation(fig, update, frames=range(10), interval=500)

# Save the GIF of the negotiation process
ani.save("output/negotiation_process.gif", writer="imagemagick", fps=2)

# Step 8: Create the HTML Dashboard with all the simulation results and visualizations
html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantum Trade Negotiation Model - Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        img {{ max-width: 100%; height: auto; }}
        h1, h2 {{ color: #333; }}
    </style>
</head>
<body>
    <h1>Quantum Trade Negotiation Model (QTNM) - Dashboard</h1>
    <p>Generated at {datetime.now().isoformat()}Z</p>

    <h2>Pareto Frontier</h2>
    <img src="pareto_frontier.png" alt="Pareto Frontier">

    <h2>Negotiation Menus</h2>
    <p>Three representative frontier packages</p>
    <img src="menus_on_frontier.png" alt="Menus on Frontier">

    <h2>Quantum Negotiation Frontier Index (Q-NFI)</h2>
    <img src="qnfi_weekly.png" alt="QNFI Weekly">

    <h2>Negotiation Process (GIF)</h2>
    <img src="negotiation_process.gif" alt="Negotiation Process">

    <h2>Frontier Head (Top 15)</h2>
    <table border="1">
        <thead>
            <tr><th>Index</th><th>US Score</th><th>CN Score</th><th>Delta T</th></tr>
        </thead>
        <tbody>
            {"".join([f"<tr><td>{i}</td><td>{front0[i]['us']}</td><td>{front0[i]['cn']}</td><td>{front0[i]['decision']['delta_t']}</td></tr>" for i in range(min(15, len(front0)))])}
        </tbody>
    </table>
</body>
</html>
"""

with open("output/dashboard.html", "w", encoding="utf-8") as f:
    f.write(html_content)

# Return the file path for the generated HTML dashboard
"output/dashboard.html"
