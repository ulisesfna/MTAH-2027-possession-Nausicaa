import json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
F = sys.argv[1]
NAVY, TEAL, ORANGE, GREY, LIGHT = "#0b2e59", "#1b8a8f", "#e07b24", "#8a94a6", "#e6ecf3"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

# Figura 1: pipeline (dos ramas independientes que solo se juntan en el stack)
fig, ax = plt.subplots(figsize=(10, 3.6)); ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")
def box(x, y, w, h, title, sub, color, txt="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.6", fc=color, ec="none", zorder=2))
    ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center", color=txt, fontsize=9.5, fontweight="bold", zorder=3)
    ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", color=txt, fontsize=7.3, zorder=3)
def arrow(x0, y0, x1, y1, lab=None, lx=None, ly=None, rad=0.0):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.5, connectionstyle=f"arc3,rad={rad}"), zorder=1)
    if lab: ax.text(lx, ly, lab, ha="center", va="center", fontsize=6.8, color="#444", style="italic", zorder=3)
# bandas de las dos ramas
ax.add_patch(FancyBboxPatch((14.5, 21.5), 37, 16, boxstyle="round,pad=0.3,rounding_size=1.5", fc="#eef3f9", ec="none", zorder=0))
ax.add_patch(FancyBboxPatch((14.5, 1.5), 37, 14, boxstyle="round,pad=0.3,rounding_size=1.5", fc="#f6efe6", ec="none", zorder=0))
ax.text(15.5, 36, "learned branch", fontsize=7.5, color=NAVY, fontweight="bold", va="center")
ax.text(15.5, 3, "rule-based branch", fontsize=7.5, color="#8a5a1a", fontweight="bold", va="center")
box(0.5, 13.5, 10.5, 10, "ATD", "ball + 22\nplayers, 25 fps", GREY)
box(16.5, 23, 13, 10, "1 · Features", "35 per frame", NAVY)
box(35, 23, 14.5, 10, "2 · TCN", "±20 s of context", NAVY)
box(31, 2.5, 14.5, 10, "3 · Rules", "who physically\ncontrols the ball", NAVY)
box(58, 13.5, 13.5, 10, "4 · Stack", "combines both\nbranches", NAVY)
box(76.5, 13.5, 11.5, 10, "5 · Viterbi", "minimum\nduration", NAVY)
box(91.5, 13.5, 8, 10, "CSV", "H / A / D", ORANGE)
arrow(11.4, 21, 16.1, 26)
arrow(11.4, 16.5, 30.6, 8)
arrow(29.9, 28, 34.6, 28)
arrow(49.9, 27, 57.6, 21.5, "H/A/D probabilities\nfrom −2 s to +2 s", 59.5, 31)
arrow(45.9, 7.5, 57.6, 15.5, "rule label, its changes\nand its confidence", 55, 5)
arrow(23, 22.6, 57.6, 18.5, "+ local geometry around the ball", 41, 15.8, rad=0.12)
arrow(71.9, 18.5, 76.1, 18.5); arrow(88.4, 18.5, 91.1, 18.5)
fig.savefig(f"{F}/fig1_pipeline.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# Figura 2: resultados por partido
d = json.load(open(sys.argv[2])); lab = [k.split("/")[1].replace("DFL-MAT-", "").replace("Sample_Game_", "Metrica ") for k in d["partits"]]
import numpy as np
x = np.arange(len(lab)); w = 0.38
fig, ax = plt.subplots(figsize=(10, 2.9))
ax.bar(x - w / 2, d["v9m1"], w, color=GREY, label=f"best model without stacking: TCN + rules + Viterbi (mean {np.mean(d['v9m1']):.3f})")
ax.bar(x + w / 2, d["ulises"], w, color=TEAL, label=f"Nausicaa (mean {np.mean(d['ulises']):.3f})")
ax.axvline(6.5, color="#bbb", lw=0.8, ls="--"); ax.text(3, 0.875, "IDSSE (Bundesliga)", ha="center", color="#555", fontsize=8); ax.text(8, 0.875, "Metrica (never trained on)", ha="center", color="#555", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=8); ax.set_ylim(0.5, 0.9); ax.set_ylabel("spell-F1")
ax.spines[["top", "right"]].set_visible(False); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8, frameon=False); ax.grid(axis="y", alpha=0.25)
fig.savefig(f"{F}/fig2_resultats.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# Figura 3: balón muerto visible en el partido (sin etiquetas)
fig, axs = plt.subplots(1, 2, figsize=(10, 2.5), gridspec_kw={"width_ratios": [1.15, 1]})
ax = axs[0]
labels = ["rules", "TCN", "stack\nbefore fix", "stack\nafter fix"]; vals = [14.8, 15.4, 9.0, 12.7]; cols = [GREY, NAVY, ORANGE, TEAL]
ax.bar(labels, vals, color=cols, width=0.6)
for i, v in enumerate(vals): ax.text(i, v + 0.3, f"{v:.1f} %", ha="center", fontsize=8)
ax.set_ylabel("% of visible-ball frames\npredicted dead (match)"); ax.set_ylim(0, 18); ax.spines[["top", "right"]].set_visible(False)
ax.set_title("Dead ball on the match (no labels used)", fontsize=9)
ax = axs[1]
cats = ["External data,\nfirst degradation", "External data,\nstage-5 degradation", "Provided\nmatch"]; med = [0.0, 0.38, 1.41]
ax.barh(cats, med, color=[GREY, ORANGE, NAVY], height=0.55)
for i, v in enumerate(med): ax.text(v + 0.03, i, f"{v:.2f} m/s", va="center", fontsize=8)
ax.set_xlabel("median ball speed when dead and visible"); ax.set_xlim(0, 1.8); ax.spines[["top", "right"]].set_visible(False)
ax.set_title("Why: dead ball stands still in external data", fontsize=9)
fig.tight_layout(); fig.savefig(f"{F}/fig3_pilota_morta.png", dpi=200, bbox_inches="tight"); plt.close(fig)
# Figura 4: historia en el mismo banco (10 partidos externos, spell-F1)
fig, ax = plt.subplots(figsize=(10, 2.6))
et = ["3 · TCN + rules\n+ Viterbi", "4a · stack with\nTCN only", "4b · stack: TCN +\nrules + geometry", "5–7 · corrected data,\nsymmetric TCN, dead ball"]
vv = [0.696, 0.705, 0.753, 0.761]; cc = [GREY, GREY, NAVY, TEAL]
ax.plot(range(4), vv, color="#c5ccd8", lw=1.5, zorder=1); ax.scatter(range(4), vv, s=90, c=cc, zorder=2)
for i, v in enumerate(vv): ax.text(i, v + 0.009, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold", color=cc[i])
ax.axhline(0.701, color=ORANGE, lw=1, ls="--"); ax.text(3.45, 0.6935, "best model without\nstacking: 0.701", va="center", fontsize=7.5, color=ORANGE)
ax.set_xticks(range(4)); ax.set_xticklabels(et, fontsize=8); ax.set_xlim(-0.4, 4.1); ax.set_ylim(0.68, 0.78); ax.set_ylabel("spell-F1, 10 external\nmatches (nested)")
ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=0.25)
fig.savefig(f"{F}/fig4_historia.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("ok")
