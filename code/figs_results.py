"""Draws the result figures from ../results (600 DPI PNG)."""
import os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sapm_env import LOSS_UNITS, ATTRS

R = os.path.join(os.path.dirname(__file__), "..", "results")
OUT = os.path.join(os.path.dirname(__file__), "..", "figs")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#E5E7EB",
                     "grid.linewidth": 0.6, "axes.edgecolor": "#9CA3AF", "axes.axisbelow": True})
AG = ["Static-Strict (DP e=0.1)", "Static-Permissive (k-anon)", "Rule-based", "Contextual bandit", "Q-learning", "SAPM-RL (proposed)"]
SHORT = ["Static-Strict", "Static-Permissive", "Rule-based", "Bandit", "Q-learning", "SAPM-RL"]
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
LS = [":", ":", "--", "-.", "-.", "-"]
INK, MUTED = "#1F2937", "#6B7280"


def save(fig, num, name):
    fig.savefig(os.path.join(OUT, f"Fig {num:02d} {name}.png"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


S = pd.read_csv(os.path.join(R, "summary_all_seeds.csv"))
G = S.groupby("agent", sort=False)
M, SD = G.mean(numeric_only=True).loc[AG], G.std(numeric_only=True).loc[AG]
C = np.load(os.path.join(R, "curves.npz"))


def fig23():
    fig, axs = plt.subplots(1, 4, figsize=(7.6, 2.9))
    for ax, k, t in zip(axs, ["final_reward", "final_utility", "final_risk", "final_violation"],
                        ["Reward (higher better)", "Utility (higher better)", "Privacy risk (lower better)", "Violation rate (lower better)"]):
        y = M[k].values; e = SD[k].values
        b = ax.barh(range(6), y, xerr=e, color=COL, height=0.62, error_kw=dict(ecolor=MUTED, lw=0.8, capsize=2))
        ax.set_yticks(range(6)); ax.set_yticklabels(SHORT if ax is axs[0] else [], fontsize=7)
        ax.invert_yaxis(); ax.set_title(t, fontsize=7.5, color=INK); ax.axvline(0, color="#9CA3AF", lw=0.8)
        ax.grid(axis="y", visible=False)
        for i, v in enumerate(y):
            ax.text(v + (0.02 * (abs(y).max() or 1)) * (1 if v >= 0 else -1), i, f"{v:.3f}", va="center",
                    ha="left" if v >= 0 else "right", fontsize=6, color=INK)
        lo = min(0, y.min()) * 1.35; hi = max(y.max(), 0) * 1.45 + 1e-3
        ax.set_xlim(lo, hi)
    fig.suptitle("Final 10,000 steps after concept drift (mean +/- s.d., 5 seeds)", fontsize=8.5, color=INK, y=1.02)
    fig.tight_layout()
    save(fig, 23, "Privacy Adaptation Performance Analysis")


def fig24():
    fig, axs = plt.subplots(2, 1, figsize=(7.4, 5.0), sharex=True)
    x = (np.arange(C[AG[0]].shape[1]) * 100 + 1000) / 1000
    for i, a in enumerate(AG):
        for ax, key in [(axs[0], a), (axs[1], a + "|viol")]:
            m, s = C[key].mean(0), C[key].std(0)
            ax.plot(x, m, color=COL[i], lw=1.6 if i == 5 else 1.1, ls=LS[i], label=SHORT[i])
            ax.fill_between(x, m - s, m + s, color=COL[i], alpha=0.12, lw=0)
    for ax in axs:
        ax.axvline(30, color=INK, lw=0.9, ls="--")
    axs[0].text(30.4, -0.35, "concept drift (t = 30k)", fontsize=7, color=INK, va="top")
    axs[0].set_ylabel("Rolling mean reward\n(window 1,000)"); axs[1].set_ylabel("Rolling violation rate")
    axs[0].set_ylim(-1.4, 0.6); axs[1].set_xlabel("Training step (thousands)")
    axs[0].legend(ncol=6, fontsize=6.5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02), handlelength=2.2)
    alarms = json.load(open(os.path.join(R, "sapm_meta.json")))["alarms"]
    for a in alarms:
        axs[0].plot(a / 1000, 0.5, marker="v", color=COL[5], ms=6)
    axs[0].text(alarms[-1] / 1000 + 0.6, 0.5, "SAPM-RL drift alarms (seed 0)", fontsize=6.5, color=INK, va="center")
    fig.tight_layout()
    save(fig, 24, "RL Convergence Assessment")


def fig25():
    T = pd.read_csv(os.path.join(R, "tradeoff.csv"))
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    sp = T[T.method == "SAPM-RL"].assign(param=lambda d: d.param.astype(float)).groupby("param").mean(numeric_only=True).reset_index()
    ax.plot(sp.risk, sp.utility, "-o", color=COL[5], lw=1.6, ms=5, label="SAPM-RL, lambda sweep (0 violations)")
    for _, r in sp[sp.param.isin([0.1, 1.2, 1.8, 2.5, 4.0])].iterrows():
        off = {0.1: (6, -12), 1.2: (2, 7), 1.8: (-10, -14), 2.5: (-62, 0), 4.0: (5, -9)}[float(r.param)]
        ax.annotate(f"lambda={float(r.param):g}", (r.risk, r.utility), textcoords="offset points", xytext=off, fontsize=6.2, color=INK)
    st = T[T.method == "Static"]
    ax.scatter(st.risk, st.utility, marker="s", s=36, color=COL[0], zorder=3, label="Static mechanism (violation rate shown)")
    for _, r in st.iterrows():
        ax.annotate(f"{r.param}\n(viol {r.violation:.2f})", (r.risk, r.utility), textcoords="offset points", xytext=(6, 2), fontsize=6.2, color=INK)
    ax.set_xlabel("Mean privacy risk per request"); ax.set_ylabel("Mean measured utility")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.set_xlim(0, 1.1); ax.set_ylim(0, 1.08)
    save(fig, 25, "Privacy-Utility Trade-Off Evaluation")


def fig26():
    labels = ["Reward", "Utility", "Low risk", "Compliance", "Post-drift\nreward", "Context\nagreement"]
    raw = np.c_[M.reward, M.utility, -M.risk, -M.violation, M.final_reward, M.final_oracle]
    nrm = (raw - raw.min(0)) / (raw.max(0) - raw.min(0) + 1e-9)
    ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist(); ang += ang[:1]
    fig, ax = plt.subplots(figsize=(5.4, 5.4), subplot_kw=dict(polar=True))
    for i in [0, 2, 4, 5]:
        v = nrm[i].tolist(); v += v[:1]
        ax.plot(ang, v, color=COL[i], lw=1.8 if i == 5 else 1.2, ls=LS[i], label=SHORT[i])
        ax.fill(ang, v, color=COL[i], alpha=0.08)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_yticks([0.25, 0.5, 0.75, 1]); ax.set_yticklabels([], fontsize=6)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.1), frameon=False, fontsize=7)
    ax.set_title("Min-max normalised across all six methods (1 = best)", fontsize=8, color=INK, pad=18)
    save(fig, 26, "Comparative Radar Chart")


def fig16():
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    t0, t1 = 45000, 48000
    for i in [1, 2, 4, 5]:
        key = AG[i].split()[0]
        a = np.load(os.path.join(R, f"actions_{key}.npy")).astype(int)
        loss = LOSS_UNITS[a[t0:t1]]
        cum = np.concatenate([np.cumsum(loss[k:k + 1000]) for k in range(0, t1 - t0, 1000)])
        ax.plot(np.arange(t0, t1) / 1000, np.minimum(cum, 2100), color=COL[i], ls=LS[i], lw=1.6 if i == 5 else 1.1, label=SHORT[i])
    ax.axhline(1200, color=INK, lw=0.9, ls="--"); ax.text(t0 / 1000 + 0.02, 1230, "window budget B = 1,200", fontsize=7, color=INK)
    ax.axhline(720, color=MUTED, lw=0.7, ls=":"); ax.text(t0 / 1000 + 0.02, 745, "'limited' level (60 %)", fontsize=6.5, color=MUTED)
    ax.set_xlabel("Step (thousands); accountant resets every 1,000 requests")
    ax.set_ylabel("Privacy-loss units requested")
    ax.legend(frameon=False, fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    save(fig, 16, "Privacy Budget Optimization Process")


def fig18():
    E = json.load(open(os.path.join(R, "explainability.json")))
    fig = plt.figure(figsize=(7.6, 4.8))
    fig.patch.set_facecolor("white")
    fig.text(0.02, 0.97, "SAPM-RL Privacy Transparency Dashboard", fontsize=10, fontweight="bold", color="#1F3A5F", va="top")
    fig.text(0.02, 0.935, "Learned policy, seed 0, after 60,000 requests", fontsize=7, color=MUTED, va="top")
    a1 = fig.add_axes([0.09, 0.50, 0.38, 0.31])
    imp = E["importance"]; ks = sorted(imp, key=imp.get)
    a1.barh(ks, [imp[k] for k in ks], color="#2a78d6", height=0.6)
    for i, k in enumerate(ks):
        a1.text(imp[k] + 0.01, i, f"{imp[k]:.2f}", va="center", fontsize=6.5, color=INK)
    a1.set_title("Attribute influence (share of decisions that flip\nwhen the attribute changes)", fontsize=7.5, color=INK)
    a1.set_xlim(0, max(imp.values()) * 1.25); a1.grid(axis="y", visible=False)
    a2 = fig.add_axes([0.58, 0.50, 0.39, 0.31])
    h = E["conf_hist"]; edges = np.linspace(0.5, 1, 11)
    a2.bar(edges[:-1] + 0.025, h, width=0.045, color="#1baf7a")
    a2.set_title(f"Decision confidence C(s), mean {E['conf_mean']:.2f}", fontsize=7.5, color=INK)
    a2.set_xlabel("C(s)", fontsize=7); a2.set_ylabel("states", fontsize=7); a2.grid(axis="x", visible=False)
    a3 = fig.add_axes([0.07, 0.08, 0.40, 0.30])
    sh = E["action_share"]
    a3.bar(list(sh.keys()), list(sh.values()), color="#eb6834", width=0.6)
    for i, v in enumerate(sh.values()):
        a3.text(i, v + 0.01, f"{v:.0%}", ha="center", fontsize=6.5, color=INK)
    a3.set_title(f"Greedy action share over {E['visited']} visited contexts", fontsize=7.5, color=INK)
    a3.tick_params(axis="x", labelsize=6.5); a3.grid(axis="x", visible=False); a3.set_ylim(0, max(sh.values()) * 1.25)
    a4 = fig.add_axes([0.55, 0.06, 0.42, 0.34]); a4.axis("off")
    a4.text(0, 0.95, "Example decision record", fontsize=7.5, fontweight="bold", color=INK, va="top")
    from sapm_env import encode, decode, ACTIONS, LEVELS, PrivacyEnv
    Q = np.load(os.path.join(R, "sapm_Q.npy"))
    ctx = [2, 1, 1, 0, 1, 0]
    def pick(c):
        q = Q[encode(c)].copy()
        mask = np.array([PrivacyEnv.violation(None, a, c) == 0 for a in range(6)]); mask[0] = True
        q[~mask] = -np.inf
        srt = np.sort(q[np.isfinite(q)])
        return int(np.argmax(q)), (1 / (1 + np.exp(-(srt[-1] - srt[-2]) / 0.1)) if len(srt) > 1 else 1.0), [ACTIONS[a] for a in range(6) if not mask[a]]
    a0, cf, blocked = pick(ctx)
    flips = []
    for i, at in enumerate(ATTRS):
        for v in range(3):
            if v != ctx[i]:
                c2 = list(ctx); c2[i] = v; a2 = pick(c2)[0]
                if a2 != a0:
                    flips.append(f"{at} -> {LEVELS[at][v].upper()} => {ACTIONS[a2]}")
    desc = ", ".join(f"{at} {LEVELS[at][ctx[i]].upper()}" for i, at in enumerate(ATTRS))
    parts = desc.split(", ")
    rec = ("Context: " + ", ".join(parts[:2]) + "\n  " + ", ".join(parts[2:4]) + "\n  " + ", ".join(parts[4:]) +
           f"\nDecision: {ACTIONS[a0]}   (confidence {cf:.2f})\nBlocked by shield: {', '.join(blocked) or 'none'}\n"
           "Would change if:\n  " + "\n  ".join(flips[:3]))
    a4.text(0, 0.82, rec, fontsize=6.0, color=INK, va="top", family="DejaVu Sans Mono")
    a4.text(0, -0.05, "Record generated from the learned Q-table (seed 0).", fontsize=6, color=MUTED)
    save(fig, 18, "Privacy Transparency Dashboard")


def fig30():
    q, rb, sp = M.loc["Q-learning"], M.loc["Rule-based"], M.loc["SAPM-RL (proposed)"]
    tiles = [("Compliance violations", f"{sp.violation:.1%}", f"vs {q.violation:.1%} for unshielded Q-learning"),
             ("Privacy risk vs Q-learning", f"-{(1 - sp.risk / q.risk):.0%}", f"{sp.risk:.3f} vs {q.risk:.3f} mean risk"),
             ("Privacy risk vs rule-based", f"-{(1 - sp.final_risk / rb.final_risk):.0%}", "after drift (final 10k steps)"),
             ("Post-drift reward vs rules", f"+{(sp.final_reward / rb.final_reward - 1):.0%}", f"{sp.final_reward:.3f} vs {rb.final_reward:.3f}")]
    fig = plt.figure(figsize=(7.6, 4.2))
    fig.text(0.02, 0.97, "Business impact dashboard (from the 5-seed experiment)", fontsize=9.5, fontweight="bold", color="#1F3A5F", va="top")
    for i, (t, v, d) in enumerate(tiles):
        ax = fig.add_axes([0.02 + i * 0.245, 0.62, 0.225, 0.27]); ax.axis("off")
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, color="#F3F4F6"))
        ax.text(0.06, 0.82, t, fontsize=7, color=MUTED, va="top")
        ax.text(0.06, 0.45, v, fontsize=17, fontweight="bold", color=INK, va="center")
        ax.text(0.06, 0.12, d, fontsize=6, color=MUTED, va="center")
    ax = fig.add_axes([0.08, 0.08, 0.86, 0.42])
    vals = M.violation.values * 1000
    ax.bar(SHORT, vals, color=COL, width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 6, f"{v:.0f}", ha="center", fontsize=7, color=INK)
    ax.set_ylabel("Non-compliant releases\nper 1,000 requests", fontsize=7.5); ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", labelsize=7)
    save(fig, 30, "Business Impact Dashboard")


if __name__ == "__main__":
    for f in [fig16, fig18, fig23, fig24, fig25, fig26, fig30]:
        f()
    print("result figures done")
