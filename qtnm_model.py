# -*- coding: utf-8 -*-
import os, json, math, random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 配置中文字体支持
plt.rcParams['font.sans-serif'] = ['Songti SC', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
import imageio  # 用于生成GIF
# from flask_babel import Babel, gettext as _

# 配置
def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)
    return p

ROOT = os.path.dirname(__file__)
OUT = ensure_dir(os.path.join(ROOT, "output"))
random.seed(2025); np.random.seed(2025)

# 多语言支持
def load_language(lang='zh'):
    translations = {
        'zh': {
            'pareto_title': "帕累托前沿（美国与中国综合分数）",
            'us_score': "美国综合得分",
            'cn_score': "中国综合得分",
            'qnfi_title': "量子谈判前沿指数 (周度)",
            'menu_title': "谈判边界前沿",
            'frontier_evolution': "QAFO优化过程 - 帕累托前沿演化",
            'conservative': "保守方案",
            'balanced': "平衡方案",
            'ambitious': "进取方案",
            'analysis_title': "量子贸易谈判模型分析报告",
            'recommendation': "建议方案",
            'tariff': "关税",
            'ntb': "非关税壁垒",
            'tech': "科技限制",
            'access': "市场准入",
            # 添加缺少的翻译项
            'delta_t': "时间框架",
            'days': "天",
            'optimization_process': "优化过程",
        },
        'en': {
            'pareto_title': "Pareto Frontier (US-China Composite Scores)",
            'us_score': "US Composite Score",
            'cn_score': "China Composite Score",
            'qnfi_title': "Quantum Negotiation Frontier Index (Weekly)",
            'menu_title': "Negotiation Boundary Frontier",
            'frontier_evolution': "QAFO Optimization - Pareto Frontier Evolution",
            'conservative': "Conservative",
            'balanced': "Balanced",
            'ambitious': "Ambitious",
            'analysis_title': "Quantum Trade Negotiation Model Analysis Report",
            'recommendation': "Recommended Package",
            'tariff': "Tariffs",
            'ntb': "Non-Tariff Barriers",
            'tech': "Tech Restrictions",
            'access': "Market Access",
            # 添加缺少的翻译项
            'delta_t': "Time Frame",
            'days': "days",
            'optimization_process': "Optimization Process",
        }
    }
    return translations.get(lang, translations['en'])

# 核心模型类
class QTNM_QAFO:
    def __init__(self, lang='zh'):
        self.lang = lang
        self.trans = load_language(lang)
        self.META = {
            "n_tau": 10, "n_ntb": 8, "n_tech": 6, "n_access": 6,
            "delta_t_options": [30, 60, 90]
        }
        self._init_coefficients()
        
    def _init_coefficients(self):
        self.Coef = {
            "us_cpi": np.random.uniform(-1.2, -0.2, size=self.META["n_tau"]),
            "us_jobs": np.random.uniform(0.1, 1.0, size=self.META["n_tau"]),
            "us_jobs_acc": np.random.uniform(0.2, 1.2, size=self.META["n_access"]),
            "us_sec": np.random.uniform(-1.0, -0.1, size=self.META["n_tech"]),
            "cn_export": np.random.uniform(0.3, 1.3, size=self.META["n_tau"]),
            "cn_tech": np.random.uniform(0.4, 1.4, size=self.META["n_tech"]),
            "cn_supply": np.random.uniform(0.2, 1.0, size=self.META["n_ntb"]),
            "ntb_cost": np.random.uniform(-0.8, -0.1, size=self.META["n_ntb"])
        }
        
        # 保存系数
        with open(os.path.join(OUT, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(self.META, f, indent=2, ensure_ascii=False)
        with open(os.path.join(OUT, "synthetic_coefficients.json"), "w", encoding="utf-8") as f:
            json.dump({k: v.tolist() for k, v in self.Coef.items()}, f, indent=2, ensure_ascii=False)
    
    def sample_decision(self):
        d = {
            "tau": np.random.randint(0, 2, size=self.META["n_tau"]),
            "ntb": np.random.randint(0, 2, size=self.META["n_ntb"]),
            "tech": np.random.randint(0, 2, size=self.META["n_tech"]),
            "access": np.random.randint(0, 2, size=self.META["n_access"]),
            "delta_t": int(np.random.choice(self.META["delta_t_options"]))
        }
        return d
    
    def check_constraints(self, d):
        if d["tech"].sum() > 3:
            return False
        if d["access"].sum() < max(1, int(0.4 * d["tau"].sum())):
            return False
        if d["tau"].sum() + d["ntb"].sum() > 12 and d["delta_t"] == 30:
            return False
        if d["tau"].sum() + d["ntb"].sum() + d["tech"].sum() + d["access"].sum() == 0:
            return False
        return True
    
    def squash_positive(self, x, scale):
        return 1.0 - math.exp(-max(0.0, x) / (scale + 1e-9))
    
    def evaluate_objectives(self, d):
        us_price = -(d["tau"] @ self.Coef["us_cpi"] + d["ntb"] @ self.Coef["ntb_cost"])
        us_jobs = d["tau"] @ self.Coef["us_jobs"] + d["access"] @ self.Coef["us_jobs_acc"]
        us_sec_pen = d["tech"] @ (-self.Coef["us_sec"])
        us_sec = max(0.0, 1.0 - (us_sec_pen / (len(self.Coef["us_sec"]) * np.abs(self.Coef["us_sec"]).mean() + 1e-9)))
        us_score = 0.4 * self.squash_positive(us_price, 8.0) + 0.4 * self.squash_positive(us_jobs, 10.0) + 0.2 * us_sec
        
        cn_exp = d["tau"] @ self.Coef["cn_export"]
        cn_tech = d["tech"] @ self.Coef["cn_tech"]
        cn_sup = d["ntb"] @ self.Coef["cn_supply"]
        cn_score = 0.4 * self.squash_positive(cn_exp, 10.0) + 0.35 * self.squash_positive(cn_tech, 8.0) + 0.25 * self.squash_positive(cn_sup, 6.0)
        
        return float(us_score), float(cn_score)
    
    def dominates(self, a, b):
        return (a[0] >= b[0] and a[1] >= b[1]) and (a[0] > b[0] or a[1] > b[1])
    
    def pareto_front(self, points):
        nd = []
        for i, p in enumerate(points):
            dom = False
            for j, q in enumerate(points):
                if i != j and self.dominates((q['us'], q['cn']), (p['us'], p['cn'])):
                    dom = True; break
            if not dom:
                nd.append(p)
        nd.sort(key=lambda x: x['us'])
        return nd
    
    def local_refine(self, sol, iters=200):
        best = dict(sol)
        for _ in range(iters):
            d = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in best["decision"].items()}
            blk = np.random.choice(["tau", "ntb", "tech", "access", "delta_t"])
            if blk == "delta_t":
                d["delta_t"] = int(np.random.choice(self.META["delta_t_options"]))
            else:
                i = np.random.randint(0, len(d[blk]))
                d[blk][i] = 1 - d[blk][i]
            if not self.check_constraints(d):
                continue
            us, cn = self.evaluate_objectives(d)
            if (us >= best["us"] and cn >= best["cn"]) and (us > best["us"] or cn > best["cn"]):
                best = {"us": us, "cn": cn, "decision": d}
        return best
    
    def run_optimization(self, n_samples=2000, refine_top=100, refine_iters=250):
        # 全局采样
        candidates = []
        for _ in range(n_samples):
            d0 = self.sample_decision()
            if not self.check_constraints(d0):
                continue
            us0, cn0 = self.evaluate_objectives(d0)
            candidates.append({"us": us0, "cn": cn0, "decision": d0})
        
        # Pareto优化过程记录（用于生成GIF）
        frames = []
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 初始前沿
        front0 = self.pareto_front(candidates)
        self._plot_frame(front0, candidates, ax, 0)
        frames.append(self._save_frame(fig))
        
        # 逐步优化
        refined = []
        for i, s in enumerate(front0[:refine_top]):
            best = self.local_refine(s, iters=refine_iters)
            refined.append(best)
            current_front = self.pareto_front(refined + front0)
            
            if i % 10 == 0:  # 每10个样本记录一帧
                self._plot_frame(current_front, candidates, ax, i+1)
                frames.append(self._save_frame(fig))
        
        # 最终前沿
        front = self.pareto_front(refined + front0)
        self._plot_frame(front, candidates, ax, "Final")
        frames.append(self._save_frame(fig))
        plt.close(fig)
        
        # 生成GIF
        gif_path = os.path.join(OUT, "qafo_optimization.gif")
        imageio.mimsave(gif_path, frames, duration=1)
        
        return front, candidates, gif_path
    
    def _plot_frame(self, front, candidates, ax, iteration):
        ax.clear()
        ax.scatter(
            [p["us"] for p in candidates], 
            [p["cn"] for p in candidates], 
            s=10, alpha=0.3, color='lightgray'
        )
        ax.scatter(
            [p["us"] for p in front], 
            [p["cn"] for p in front], 
            s=30, color='blue'
        )
        ax.set_xlabel(self.trans['us_score'])
        ax.set_ylabel(self.trans['cn_score'])
        title = f"{self.trans['frontier_evolution']} - "
        title += f"{'迭代' if self.lang == 'zh' else 'Iteration'} {iteration}" if isinstance(iteration, int) else ("最终结果" if self.lang == 'zh' else "Final Result")
        ax.set_title(title)
        ax.grid(True)
    
    def _save_frame(self, fig):
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return image
    
    def decisions_to_df(self, points):
        rows = []
        for i, p in enumerate(points):
            d = p["decision"]
            rows.append({
                "idx": i, "us_score": p["us"], "cn_score": p["cn"], "delta_t": d["delta_t"],
                **{f"tau_{k}": int(v) for k, v in enumerate(d["tau"])},
                **{f"ntb_{k}": int(v) for k, v in enumerate(d["ntb"])},
                **{f"tech_{k}": int(v) for k, v in enumerate(d["tech"])},
                **{f"access_{k}": int(v) for k, v in enumerate(d["access"])},
            })
        return pd.DataFrame(rows)
    
    def pick_menus(self, front):
        if len(front) == 0:
            return []
        if len(front) <= 3:
            return front
        return [front[0], front[len(front)//2], front[-1]]
    
    def generate_qnfi(self, front):
        qnfi_val = max(min(p["us"], p["cn"]) for p in front) if front else 0.0
        weeks = [datetime.today() - timedelta(days=7*i) for i in range(12, -1, -1)]
        weeks_str = [w.strftime("%Y-%m-%d") for w in weeks]
        hist = [max(0.0, min(1.0, qnfi_val + np.random.normal(0.0, 0.05) + 0.02*(i-6))) for i in range(len(weeks))]
        return pd.DataFrame({"week": weeks_str, "QNFI": hist})
    
    def generate_all_outputs(self):
        # 运行优化
        front, candidates, gif_path = self.run_optimization()
        
        # 保存前沿决策
        df_front = self.decisions_to_df(front)
        df_front.to_csv(os.path.join(OUT, "pareto_frontier_decisions.csv"), index=False)
        
        # 生成谈判菜单
        menus = self.pick_menus(front)
        df_menus = self.decisions_to_df(menus)
        df_menus.to_csv(os.path.join(OUT, "negotiation_menus.csv"), index=False)
        
        # 生成Q-NFI时间序列
        df_qnfi = self.generate_qnfi(front)
        df_qnfi.to_csv(os.path.join(OUT, "qnfi_timeseries.csv"), index=False)
        
        # 生成图表
        self._generate_charts(front, candidates, menus, df_qnfi)
        
        # 生成报告
        self._generate_reports(menus, df_front)
        
        return {
            "front": front,
            "menus": menus,
            "qnfi": df_qnfi,
            "gif_path": gif_path
        }
    
    def _generate_charts(self, front, candidates, menus, df_qnfi):
        # 帕累托前沿图
        plt.figure(figsize=(8, 6))
        plt.scatter([p["us"] for p in candidates], [p["cn"] for p in candidates], s=10, alpha=0.3, label="候选方案" if self.lang == 'zh' else "Candidate Solutions")
        plt.scatter([p["us"] for p in front], [p["cn"] for p in front], s=30, color='blue', label="帕累托前沿" if self.lang == 'zh' else "Pareto Frontier")
        plt.xlabel(self.trans['us_score'])
        plt.ylabel(self.trans['cn_score'])
        plt.title(self.trans['pareto_title'])
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, "pareto_frontier.png"), dpi=160)
        plt.close()
        
        # 菜单在帕累托前沿上的位置
        plt.figure(figsize=(8, 6))
        plt.scatter([p["us"] for p in front], [p["cn"] for p in front], s=18, alpha=0.7)
        menu_labels = [self.trans['conservative'], self.trans['balanced'], self.trans['ambitious']]
        colors = ['green', 'blue', 'red']
        for m, label, color in zip(menus, menu_labels[:len(menus)], colors):
            plt.scatter([m["us"]], [m["cn"]], s=90, color=color)
            plt.text(m["us"], m["cn"]+0.01, label, fontsize=12, ha='center')
        plt.xlabel(self.trans['us_score'])
        plt.ylabel(self.trans['cn_score'])
        plt.title(self.trans['menu_title'])
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, "menus_on_frontier.png"), dpi=160)
        plt.close()
        
        # Q-NFI时间序列图
        plt.figure(figsize=(10, 5))
        plt.plot(df_qnfi['week'], df_qnfi['QNFI'], marker='o', linestyle='-')
        plt.xticks(rotation=45)
        plt.ylabel("Q-NFI")
        plt.title(self.trans['qnfi_title'])
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, "qnfi_weekly.png"), dpi=160)
        plt.close()
    
    def _generate_reports(self, menus, df_front):
        # 生成HTML报告
        html_report = self._generate_html_report(menus)
        with open(os.path.join(OUT, "negotiation_report.html"), "w", encoding="utf-8") as f:
            f.write(html_report)
        
        # 生成Markdown报告模板
        md_report = self._generate_md_report(menus)
        with open(os.path.join(OUT, "negotiation_report_template.md"), "w", encoding="utf-8") as f:
            f.write(md_report)
    
    def _generate_html_report(self, menus):
        # 报告生成逻辑（简化版）
        report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{self.trans['analysis_title']}</title>
            <style>/* 样式省略 */</style>
        </head>
        <body>
            <h1>{self.trans['analysis_title']}</h1>
            <h2>{self.trans['recommendation']}</h2>
            <!-- 菜单详情表格 -->
            
            
            <!-- 详细分析 -->
        </body>
        </html>
        """
        return report
    
    def _generate_md_report(self, menus):
        # 简化版Markdown报告
        return f"""
        # {self.trans['analysis_title']}
        ## {self.trans['recommendation']}
        
        ### {self.trans['conservative']}
        - **US得分**: {menus[0]['us']:.3f}
        - **CN得分**: {menus[0]['cn']:.3f}
        - **时间框架**: {menus[0]['decision']['delta_t']}天
        
        ### {self.trans['balanced']}
        - **US得分**: {menus[1]['us']:.3f}
        - **CN得分**: {menus[1]['cn']:.3f}
        - **时间框架**: {menus[1]['decision']['delta_t']}天
        
        ### {self.trans['ambitious']}
        - **US得分**: {menus[2]['us']:.3f}
        - **CN得分**: {menus[2]['cn']:.3f}
        - **时间框架**: {menus[2]['decision']['delta_t']}天
        """