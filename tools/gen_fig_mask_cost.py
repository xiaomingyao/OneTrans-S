#!/usr/bin/env python3
# coding=utf-8
"""迁移报告配图：mask 形态对 attention kernel 与端到端步时的影响"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for f in ("Noto Sans CJK SC", "Noto Sans SC", "WenQuanYi Zen Hei"):
    if any(f.lower() in x.name.lower() for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [f]
        break
plt.rcParams["axes.unicode_minus"] = False

RED, ACC, GRAY, INK = "#C00000", "#2b6b8a", "#9aa4b2", "#1f2733"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.6))
fig.suptitle("OneTrans 迁 950：attention mask 形态的代价（实测）", fontsize=13.5,
             fontweight="bold", color=RED, y=0.97)

# (a) attention 单次耗时 + kernel 数
labels = ["float 加性 mask\n（参考实现默认）", "bool mask\n（迁移建议）", "无 mask\n（上限参考）"]
t_us = [46.3, 12.3, 9.3]
n_k = [45, 6, 3]
b = ax1.bar(range(3), t_us, color=[GRAY, ACC, "#b7c9d6"], width=0.6)
for i, (t, k) in enumerate(zip(t_us, n_k)):
    ax1.text(i, t + 1.2, f"{t:.1f} µs\n{k} 个 kernel", ha="center", fontsize=9.5,
             color=INK, fontweight="bold")
ax1.set_xticks(range(3)); ax1.set_xticklabels(labels, fontsize=8.8)
ax1.set_ylabel("单次 attention 耗时（µs）", fontsize=9.5)
ax1.set_ylim(0, 56)
ax1.grid(axis="y", ls=":", color="#dfe4ea"); ax1.set_axisbelow(True)
for sp in ("top", "right"): ax1.spines[sp].set_visible(False)
ax1.set_title("(a) float 掩码退回手工路径：Cast+Muls+MatMul+Softmax，46.3 µs", fontsize=10.5, loc="left")

# (b) 端到端步时
cfg = ["B=64, S=64", "B=32, S=256"]
fl = [55.9, 55.8]
bl = [55.0, 53.8]
x = range(len(cfg)); w = 0.34
ax2.bar([i - w/2 for i in x], fl, width=w, color=GRAY, label="float mask")
ax2.bar([i + w/2 for i in x], bl, width=w, color=ACC, label="bool mask")
for i, (a, b2) in enumerate(zip(fl, bl)):
    ax2.text(i - w/2, a + 0.5, f"{a:.1f}", ha="center", fontsize=9, color=INK)
    ax2.text(i + w/2, b2 + 0.5, f"{b2:.1f}", ha="center", fontsize=9, color=INK)
    ax2.text(i, min(a, b2) - 3.2, f"{(a-b2)/a*100:+.1f}%", ha="center", fontsize=10.5,
             color=RED, fontweight="bold")
ax2.set_xticks(list(x)); ax2.set_xticklabels(cfg, fontsize=9.5)
ax2.set_ylabel("训练步时（ms，20 次中位）", fontsize=9.5)
ax2.set_ylim(44, 60)
ax2.grid(axis="y", ls=":", color="#dfe4ea"); ax2.set_axisbelow(True)
ax2.legend(frameon=False, fontsize=9, loc="lower right")
for sp in ("top", "right"): ax2.spines[sp].set_visible(False)
ax2.set_title("(b) 端到端只 +1.6% / +3.7%：成本在多组小 FFN，不在 attention", fontsize=10.5, loc="left")

fig.tight_layout(rect=(0, 0, 1, 0.93))
out = "docs/figures/fig_mask_cost.png"
fig.savefig(out, dpi=140)
print("saved", out)
