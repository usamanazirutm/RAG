"""Generate the two result figures from results/table4.csv and table5.csv.

  fig_asr_by_class.png   grouped bars: baseline vs hardened ASR per attack class (with Wilson CIs)
  fig_ablation.png       leave-one-out ablation: overall ASR and OT-write rate

Run after run_experiment.py:  python make_figures.py
"""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "results"


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fig_asr_by_class():
    rows = [r for r in read_csv(os.path.join(OUT, "table4.csv")) if r["class"] != "ALL"]
    cls = [r["class"] for r in rows]
    base = np.array([float(r["ASR_baseline"]) for r in rows])
    hard = np.array([float(r["ASR_hardened"]) for r in rows])
    base_err = [base - np.array([float(r["ASR_base_lo"]) for r in rows]),
                np.array([float(r["ASR_base_hi"]) for r in rows]) - base]
    hard_err = [hard - np.array([float(r["ASR_hard_lo"]) for r in rows]),
                np.array([float(r["ASR_hard_hi"]) for r in rows]) - hard]
    x = np.arange(len(cls)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - w/2, base, w, yerr=base_err, capsize=3, label="Baseline", color="#c0392b")
    ax.bar(x + w/2, hard, w, yerr=hard_err, capsize=3, label="Hardened (M1-M7)", color="#2471a3")
    ax.set_xticks(x); ax.set_xticklabels(cls)
    ax.set_ylabel("Attack success rate (%)"); ax.set_ylim(0, 105)
    ax.set_xlabel("Attack class")
    ax.set_title("Attack success by class, baseline vs hardened")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_asr_by_class.png"), dpi=200)
    plt.close(fig)


def fig_ablation():
    rows = read_csv(os.path.join(OUT, "table5.csv"))
    order = ["baseline", "hardened", "-M1", "-M2", "-M3", "-M4", "-M5", "-M6", "-M7"]
    rows = sorted(rows, key=lambda r: order.index(r["config"]) if r["config"] in order else 99)
    labels = [r["config"] for r in rows]
    asr = [float(r["overall_ASR"]) for r in rows]
    otw = [float(r["OTwrite_rate"]) for r in rows]
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(x - w/2, asr, w, label="Overall ASR (%)", color="#884ea0")
    ax.bar(x + w/2, otw, w, label="OT-write rate (%)", color="#d68910")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("%"); ax.set_title("Leave-one-out mitigation ablation")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_ablation.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig_asr_by_class()
    fig_ablation()
    print("Wrote results/fig_asr_by_class.png and results/fig_ablation.png")
