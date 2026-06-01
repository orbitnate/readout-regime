#!/usr/bin/env python3
"""Generate figures for the Readout-Regime paper.

Figure 2 (cosine-dosage / off-axis attractor) uses the verified numbers from
PITWM/docs/results/BASIS_MECHANISM_PROBE_RESULTS.md (section D-iii, mult=4,
3-5 seed medians). It shows: (a) Delta-logp is ~linear in cos(A, lm_head[t]) on
the base model (Pythia-1.4B), and (b) on instruction-tuned/chat models an
off-axis residual attractor lifts a norm-matched random install (cos=0) and makes
the dosage non-monotone on Llama (cos=0.7 exceeds canonical cos=1.0).

Run:  python make_figures.py   ->  fig_attractor.pdf / .png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

cos = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
series = {
    "Pythia-1.4B (base)":      [0.16, 1.77, 5.08, 8.22, 10.97, 13.03],
    "Mistral-7B-Instruct":     [1.0,  3.4,  9.6,  14.1, 16.0,  15.6],
    "Llama-3.1-8B-Instruct":   [2.5,  7.2,  14.6, 18.8, 20.4,  13.9],
    "Qwen2.5-7B-Instruct":     [7.4,  18.0, 18.5, 18.5, 18.5,  18.5],
}
markers = {"Pythia-1.4B (base)": "o", "Mistral-7B-Instruct": "s",
           "Llama-3.1-8B-Instruct": "^", "Qwen2.5-7B-Instruct": "D"}

fig, ax = plt.subplots(figsize=(6.0, 4.2))
for name, ys in series.items():
    ax.plot(cos, ys, marker=markers[name], linewidth=1.8, markersize=5, label=name)
ax.set_xlabel(r"$\cos(A,\ \mathrm{lm\_head}[t])$")
ax.set_ylabel(r"$\Delta\,\log p$(target)  [nats, mult=4]")
ax.set_title("Cosine-dosage of the install direction (mult = 4)", fontsize=11)
ax.set_ylim(-1, 24)
ax.axhline(0.0, color="0.7", linewidth=0.8, zorder=0)
ax.annotate("random install (cos=0):\n~0 on base, lifted on instruct", xy=(0.0, 7.4),
            xytext=(0.08, 4.0), fontsize=8,
            arrowprops=dict(arrowstyle="->", color="0.5"))
ax.annotate("Llama non-monotone:\ncos=0.7 (20.4) > cos=1.0 (13.9)", xy=(0.7, 20.4),
            xytext=(0.20, 22.6), fontsize=8,
            arrowprops=dict(arrowstyle="->", color="0.5"))
ax.legend(fontsize=8, loc="lower right")
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig("fig_attractor.pdf")
fig.savefig("fig_attractor.png", dpi=160)
print("wrote fig_attractor.pdf / fig_attractor.png")
