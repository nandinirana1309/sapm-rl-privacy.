"""Draws the conceptual / architecture figures (600 DPI PNG)."""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Wedge
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "figs")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})

NAVY, TEAL, AMBER, RED, GREY, GREEN, PURPLE = "#1F3A5F", "#2A9D8F", "#E9A23B", "#C8553D", "#6B7280", "#4C9A5B", "#6D5BA8"
LIGHT = {NAVY: "#DCE6F2", TEAL: "#D7F0EC", AMBER: "#FBEBD0", RED: "#F6DCD6", GREY: "#ECEEF1", GREEN: "#DDEFE0", PURPLE: "#E6E1F3"}


def canvas(w=7.5, h=4.4, xmax=100, ymax=60):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, xmax); ax.set_ylim(0, ymax); ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, c=NAVY, fs=7.5, bold=False, fill=None, tc="#111827", r=1.2):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={r}",
                       linewidth=1.0, edgecolor=c, facecolor=fill or LIGHT.get(c, "#FFFFFF"))
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal", wrap=True, linespacing=1.25)
    return (x, y, w, h)


def header(ax, x, y, w, h, text, c=NAVY, fs=8):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=1.0", linewidth=0, facecolor=c)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color="white", fontweight="bold")


def frame(ax, x, y, w, h, label, c=GREY):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=1.5", linewidth=1.0,
                       edgecolor=c, facecolor="none", linestyle=(0, (4, 3)))
    ax.add_patch(p)
    ax.text(x + 1.2, y + h - 1.6, label, ha="left", va="top", fontsize=7, color=c, fontweight="bold")


def arrow(ax, p1, p2, label=None, c="#374151", style="-|>", ls="-", rad=0.0, fs=6.5, lw=1.0):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=9, color=c, linewidth=lw,
                        linestyle=ls, connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)
    if label:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my + 0.9, label, fontsize=fs, ha="center", va="bottom", color=c,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.4))


def mid(b, side):
    x, y, w, h = b
    return {"l": (x, y + h / 2), "r": (x + w, y + h / 2), "t": (x + w / 2, y + h), "b": (x + w / 2, y)}[side]


def save(fig, num, name):
    fn = os.path.join(OUT, f"Fig {num:02d} {name}.png")
    fig.savefig(fn, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 1
def fig1():
    fig, ax = canvas(h=3.4, ymax=46)
    eras = [("1995-2005", "Notice & consent\nstatic policies,\nP3P, opt-in/out", GREY),
            ("2000-2012", "Rule-based\naccess control\nRBAC/XACML/ABAC", NAVY),
            ("2006-2018", "Privacy-enhancing\ntech: k-anonymity,\ndifferential privacy", TEAL),
            ("2018-2023", "Compliance\nautomation: GDPR,\nCCPA, DPDP Act", AMBER),
            ("2020 onward", "Self-adaptive\nprivacy: context-\naware RL agents", GREEN)]
    ax.plot([4, 96], [26, 26], color="#9CA3AF", lw=2)
    for i, (yr, txt, c) in enumerate(eras):
        x = 4 + i * 19
        ax.add_patch(Circle((x + 8, 26), 1.3, color=c, zorder=3))
        ax.text(x + 8, 30, yr, ha="center", fontsize=7.5, fontweight="bold", color=c)
        box(ax, x, 6, 17.5, 15, txt, c, fs=5.7)
        arrow(ax, (x + 8, 24.6), (x + 8, 21.2), c=c)
    ax.text(50, 40, "Increasing context awareness, automation and adaptivity", ha="center", fontsize=8, color=NAVY, style="italic")
    arrow(ax, (12, 36.5), (88, 36.5), c=NAVY, lw=1.6)
    save(fig, 1, "Evolution of Privacy Management Approaches")


# ---------------------------------------------------------------- Fig 2
def fig2():
    fig, ax = canvas(h=4.6, ymax=62)
    core = box(ax, 37, 24, 26, 14, "SAPM-RL\nSelf-Adaptive Privacy\nManagement Core", GREEN, fs=8.5, bold=True)
    nodes = [("Data subjects\nconsent & preferences", 4, 46, TEAL), ("Data requesters\nclinicians, analysts, apps", 70, 46, NAVY),
             ("Regulators & DPO\nGDPR, HIPAA, DPDP", 4, 4, AMBER), ("Security operations\nSIEM, threat intel", 70, 4, RED),
             ("Data stores\nEHR, IoT, logs", 4, 25, GREY), ("Audit & transparency\nportal, reports", 70, 25, PURPLE)]
    for t, x, y, c in nodes:
        b = box(ax, x, y, 26, 11, t, c, fs=7)
        cx, cy = x + 13, y + 5.5
        tx = 37 if cx < 50 else 63
        ty = min(max(cy, 26), 36)
        arrow(ax, (x + 26 if cx < 50 else x, cy), (tx, ty), c=c, style="<|-|>")
    ax.text(50, 58, "Adaptive Privacy Management Ecosystem", ha="center", fontsize=9, fontweight="bold", color=NAVY)
    save(fig, 2, "Adaptive Privacy Management Ecosystem")


# ---------------------------------------------------------------- Fig 3
def fig3():
    fig, ax = canvas(h=4.4)
    ax.add_patch(Circle((50, 30), 9, color=LIGHT[RED], ec=RED))
    ax.text(50, 30, "Dynamic\nprivacy\nproblem", ha="center", va="center", fontsize=8.5, fontweight="bold", color=RED)
    items = [("Evolving threats\ninsider misuse, linkage,\nmembership inference", 0), ("Shifting user\npreferences & consent\nwithdrawal", 60),
             ("Regulatory change\nnew rules, fines,\ncross-border limits", 120), ("Context-dependent\naccess: who, why,\nwhen, from where", 180),
             ("Privacy-utility\ntension: noise vs\nanalytic value", 240), ("Finite privacy\nbudget exhausted\nby repeated queries", 300)]
    for t, ang in items:
        r = np.deg2rad(ang + 90)
        cx, cy = 50 + 33 * np.cos(r), 30 + 21 * np.sin(r)
        box(ax, cx - 12, cy - 5.5, 24, 11, t, [NAVY, TEAL, AMBER, PURPLE, GREEN, GREY][items.index((t, ang))], fs=6.6)
        arrow(ax, (cx - (cx - 50) * 0.42, cy - (cy - 30) * 0.42), (50 + (cx - 50) * 0.27, 30 + (cy - 30) * 0.40), c=GREY)
    save(fig, 3, "Dynamic Privacy Challenge Landscape")


# ---------------------------------------------------------------- Fig 4
def fig4():
    fig, ax = canvas(h=4.8, ymax=64)
    root = box(ax, 33, 54, 34, 7, "Privacy management research", NAVY, fs=8.5, bold=True)
    cats = [("Privacy\nengineering", NAVY, ["Privacy by Design", "Design strategies", "Policy NLP / audit"]),
            ("Privacy-enhancing\ntechnologies", TEAL, ["Differential privacy", "k-anon / l-div / t-close", "FL, HE, SMC"]),
            ("Access control", AMBER, ["RBAC / XACML", "ABAC, context-aware", "Risk-adaptive AC"]),
            ("Self-adaptive\nsystems", PURPLE, ["MAPE-K loops", "Adaptive security", "Privacy awareness"]),
            ("Reinforcement\nlearning", GREEN, ["Q-learning, DQN, PPO", "Safe / constrained RL", "MARL, RL for security"]),
            ("Trustworthy &\nexplainable AI", RED, ["XAI, XRL", "Fairness, accountability", "AI governance"])]
    for i, (t, c, subs) in enumerate(cats):
        x = 1 + i * 16.5
        b = box(ax, x, 38, 16, 9, t, c, fs=6.2, bold=True)
        arrow(ax, (50, 54), (x + 7.5, 47), c=GREY)
        for j, s in enumerate(subs):
            sb = box(ax, x + 0.2, 27 - j * 9, 15.6, 6.5, s.replace(" / ", "/\n").replace(", ", ",\n"), c, fs=5.6, fill="white")
            arrow(ax, (x + 7.5, 38 if j == 0 else 27 - (j - 1) * 9), (x + 7.5, 33.5 - j * 9), c=c)
    ax.text(50, 3, "Gap addressed in this study: the intersection of RL + PETs + access control + governance + XAI",
            ha="center", fontsize=7.2, color=NAVY, style="italic")
    save(fig, 4, "Literature Taxonomy Framework")


# ---------------------------------------------------------------- Fig 5
def fig5():
    fig, ax = canvas(h=3.8, ymax=52)
    src = [("Regulations\nGDPR, HIPAA, DPDP", AMBER), ("Stakeholder goals\nsubjects, DPO, analysts", TEAL), ("Threat models\nlinkage, inference, insider", RED)]
    for i, (t, c) in enumerate(src):
        box(ax, 1, 38 - i * 15, 19, 11, t, c, fs=6.0)
        arrow(ax, (20, 43.5 - i * 15), (25, 28), c=c)
    e = box(ax, 25, 21, 15, 14, "Requirements\nelicitation &\nprioritisation", NAVY, fs=7, bold=True)
    fr = box(ax, 46, 30, 18, 13, "Functional\n(FR1-FR10): sense,\ndecide, enforce,\nexplain, audit", GREEN, fs=6.0)
    nfr = box(ax, 46, 11, 18, 13, "Non-functional\n(NFR1-NFR10):\nlatency, scalability,\nrobustness, fairness", PURPLE, fs=6.0)
    arrow(ax, (40, 30), (46, 36.5), c=NAVY); arrow(ax, (40, 26), (46, 17.5), c=NAVY)
    v = box(ax, 70, 20, 13, 16, "Acceptance\ncriteria &\nmetrics\n(Sec. 15)", TEAL, fs=6.6)
    arrow(ax, (64, 36.5), (70, 30), c=GREY); arrow(ax, (64, 17.5), (70, 26), c=GREY)
    t = box(ax, 87, 20, 12, 16, "Trace to\narchitecture\nlayers\n(Sec. 5-13)", AMBER, fs=6.6)
    arrow(ax, (83, 28), (87, 28), c=GREY)
    arrow(ax, (93, 20), (32, 21), label=None, c=GREY, ls="--", rad=-0.5)
    save(fig, 5, "Requirements Engineering Framework")


# ---------------------------------------------------------------- Fig 6  (core)
def fig6():
    fig, ax = canvas(w=7.8, h=6.0, ymax=80)
    ax.text(50, 78, "SAPM-RL: Self-Adaptive Privacy Management Architecture", ha="center", fontsize=9.5, fontweight="bold", color=NAVY)
    # left sources
    srcs = ["Access requests", "IAM / roles", "Consent registry", "SIEM & threat feeds", "Data catalogue", "Budget accountant"]
    frame(ax, 0.5, 8, 17, 64, "Sources", GREY)
    for i, s in enumerate(srcs):
        box(ax, 2, 62 - i * 9.5, 14, 7, s.replace(" & ", " &\n").replace(" accountant", "\naccountant"), GREY, fs=5.8)
    # monitor
    m = box(ax, 22, 52, 18, 16, "MONITOR\nContext Modelling\nLayer (Sec. 6)\nstate s_t", TEAL, fs=6.2, bold=True)
    a = box(ax, 22, 28, 18, 16, "ANALYSE\nRisk Assessment\nLayer (Sec. 9)\nRisk(s,a)", RED, fs=6.2, bold=True)
    p = box(ax, 45, 28, 22, 40, "", GREEN)
    ax.text(56, 64.5, "PLAN", ha="center", fontsize=7.5, fontweight="bold", color=GREEN)
    box(ax, 46.5, 54, 19, 8, "RL Privacy Intelligence\nQ-learning agent (Sec. 7)", GREEN, fs=5.8, fill="white")
    box(ax, 46.5, 44, 19, 8, "Compliance shield\n(action mask, Sec. 12)", AMBER, fs=5.8, fill="white")
    box(ax, 46.5, 34, 19, 8, "Lagrangian risk control\n+ drift detector", PURPLE, fs=5.8, fill="white")
    ax.text(56, 30.2, "Knowledge: Q-table, policies,\nrisk targets, audit history", ha="center", fontsize=6.0, color=GREY)
    e = box(ax, 72, 46, 18, 22, "EXECUTE\nDynamic Policy\nAdaptation (Sec. 8)\n+ DP / k-anon /\nencryption (Sec. 10)", NAVY, fs=6.0, bold=True)
    x = box(ax, 72, 28, 18, 14, "Explainability\nLayer (Sec. 11)\nreasons, confidence", PURPLE, fs=6.0)
    out = box(ax, 93, 30, 6.5, 34, "Pro-\ntected\nrelease", GREY, fs=5.6)
    for i in range(6):
        arrow(ax, (16, 65.5 - i * 9.5), (22, 60), c="#9CA3AF", lw=0.7)
    arrow(ax, mid(m, "b"), mid(a, "t"), c=TEAL)
    arrow(ax, (40, 60), (47, 58), c=TEAL, label="s_t")
    arrow(ax, (40, 36), (47, 38), c=RED, label="risk")
    arrow(ax, (65, 58), (72, 58), c=GREEN, label="a_t")
    arrow(ax, (81, 46), (81, 42), c=NAVY)
    arrow(ax, (90, 57), (93, 50), c=NAVY)
    # bottom layers
    g = box(ax, 22, 12, 34, 10, "Governance, Compliance & Enforcement\n(Sec. 12): rules, audit log, DPO", AMBER, fs=5.9)
    s = box(ax, 59, 12, 31, 10, "Security & Threat Intelligence\n(Sec. 13): anomalies, threat level", RED, fs=5.9)
    arrow(ax, (39, 22), (52, 44), c=AMBER, ls="--")
    arrow(ax, (74, 22), (31, 52), c=RED, ls="--", rad=0.15)
    arrow(ax, (99, 30), (99, 4), c=GREY, ls="--", style="-")
    arrow(ax, (99, 4), (8, 4), c=GREY, ls="--", style="-")
    arrow(ax, (8, 4), (8, 8), c=GREY, ls="--", label=None)
    ax.text(54, 5.3, "Feedback: measured utility, privacy loss, violations, incidents -> reward r_t (Eq. 3)", ha="center", fontsize=6.5, color=GREY)
    save(fig, 6, "Self-Adaptive Privacy Management Architecture")


# ---------------------------------------------------------------- Fig 7
def fig7():
    fig, ax = canvas(h=5.0, ymax=66)
    layers = [("L8  Security & Threat Intelligence", "anomaly detection, threat feeds, incident response", RED),
              ("L7  Governance, Compliance & Enforcement", "policy-as-code, compliance shield, audit trail", AMBER),
              ("L6  Explainability & Transparency", "decision reasons, confidence, dashboards", PURPLE),
              ("L5  Differential Privacy & Data Protection", "Laplace / Gaussian, k-anon, encryption, accountant", TEAL),
              ("L4  Privacy Risk Assessment & Decision", "leakage x sensitivity x threat x trust", RED),
              ("L3  Dynamic Policy Adaptation", "policy generation, conflict resolution, versioning", NAVY),
              ("L2  RL Privacy Intelligence", "state, action, reward, Q-learning, drift handling", GREEN),
              ("L1  Data Acquisition & Context Modelling", "requests, IAM, consent, SIEM, catalogue", GREY)]
    for i, (t, d, c) in enumerate(layers):
        y = 58 - i * 7.2
        header(ax, 1, y, 41, 6, t, c, fs=6.2)
        box(ax, 42, y, 46, 6, d, c, fs=6.8)
    ax.annotate("", xy=(94, 59), xytext=(94, 4), arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.4))
    ax.text(96.5, 32, "context & events flow up", rotation=90, va="center", fontsize=6.5, color=NAVY)
    ax.annotate("", xy=(91, 4), xytext=(91, 59), arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.4))
    ax.text(89, 32, "decisions flow down", rotation=90, va="center", fontsize=6.5, color=GREEN)
    save(fig, 7, "Multi-Layer Privacy Intelligence Architecture")


# ---------------------------------------------------------------- Fig 8
def fig8():
    fig, ax = canvas(h=4.0, ymax=54)
    srcs = [("Request metadata", "attribute, cohort, purpose"), ("IAM / directory", "role, clearance -> trust"),
            ("Consent registry", "subject preference"), ("SIEM / IDS", "events -> threat level"),
            ("Data catalogue", "attribute sensitivity class"), ("Privacy accountant", "loss spent in window")]
    for i, (t, d) in enumerate(srcs):
        box(ax, 1, 46 - i * 8.2, 22, 6.6, f"{t}\n{d}", GREY, fs=6.2)
        arrow(ax, (23, 49.3 - i * 8.2), (29, 27), c="#9CA3AF", lw=0.7)
    st = [("Collect & validate", TEAL), ("Normalise &\nde-duplicate", TEAL), ("Discretise into\n3 levels", NAVY), ("Encode context\nvector s_t", GREEN)]
    for i, (t, c) in enumerate(st):
        b = box(ax, 29 + i * 17, 21, 14, 12, t, c, fs=6.8)
        if i:
            arrow(ax, (29 + i * 17 - 3, 27), (29 + i * 17, 27), c=GREY)
    ax.text(84, 14, "s_t = (sensitivity, trust, threat,\npurpose, consent, budget)\n|S| = 3^6 = 729", ha="center", fontsize=6.8, color=GREEN)
    arrow(ax, (84, 33), (84, 44), c=GREEN, label="to RL agent")
    save(fig, 8, "Privacy Context Acquisition Framework")


# ---------------------------------------------------------------- Fig 9
def fig9():
    fig, ax = canvas(h=4.2, ymax=56)
    ag = box(ax, 33, 33, 34, 18, "", GREEN)
    ax.text(50, 48.5, "RL Privacy Engine (agent)", ha="center", fontsize=8, fontweight="bold", color=GREEN)
    box(ax, 35, 35, 9.5, 10, "Q-table\nQ(s,a)", GREEN, fs=6.5, fill="white")
    box(ax, 45.5, 35, 9.5, 10, "Policy\nshielded\ne-greedy", GREEN, fs=6.5, fill="white")
    box(ax, 56.5, 35, 9.5, 10, "Learner\nTD update,\ndual mu", GREEN, fs=6.5, fill="white")
    env = box(ax, 33, 4, 34, 15, "Privacy environment\nrequests on real health data,\nthreat dynamics, consent, regulations", NAVY, fs=7)
    arrow(ax, (67, 42), (85, 42), c=GREEN, style="-")
    arrow(ax, (85, 42), (85, 11.5), c=GREEN, style="-")
    arrow(ax, (85, 11.5), (67, 11.5), c=GREEN, label="action a_t (mechanism)")
    arrow(ax, (33, 14), (15, 14), c=NAVY, style="-")
    arrow(ax, (15, 14), (15, 42), c=NAVY, style="-")
    arrow(ax, (15, 42), (33, 42), c=NAVY, label="state s_t+1")
    arrow(ax, (33, 8), (8, 8), c=AMBER, style="-")
    arrow(ax, (8, 8), (8, 38), c=AMBER, style="-")
    arrow(ax, (8, 38), (33, 38), c=AMBER, label="reward r_t")
    save(fig, 9, "Reinforcement Learning Privacy Engine")


# ---------------------------------------------------------------- Fig 10
def fig10():
    fig, ax = canvas(h=3.4, ymax=44)
    steps = [("1 Observe\ncontext s_t", TEAL), ("2 Mask\nnon-compliant\nactions", AMBER), ("3 Select a_t\n(e-greedy)", GREEN),
             ("4 Apply\nmechanism\non data", NAVY), ("5 Measure\nU, Risk, V", RED), ("6 Reward\nr_t (Eq. 3)", AMBER),
             ("7 Update Q\n(Eq. 4) & mu\n(Eq. 6)", GREEN), ("8 Drift test\n(Eq. 7)", PURPLE)]
    for i, (t, c) in enumerate(steps):
        x = 1 + (i % 4) * 25 if i < 4 else 1 + (7 - i) * 25
        y = 26 if i < 4 else 4
        box(ax, x, y, 20, 13, t, c, fs=6.8)
        if i % 4:
            if i < 4:
                arrow(ax, (x - 5, y + 6.5), (x, y + 6.5), c=GREY)
            else:
                pass
    for k in range(3):
        x = 1 + k * 25
        arrow(ax, (x + 25, 10.5), (x + 20, 10.5), c=GREY)
    arrow(ax, (86, 26), (86, 17), c=GREY)
    arrow(ax, (11, 17), (11, 26), c=GREY, ls="--")
    save(fig, 10, "State-Action-Reward Workflow")


# ---------------------------------------------------------------- Fig 11
def fig11():
    fig, ax = canvas(h=4.0, ymax=54)
    rp = box(ax, 2, 34, 22, 14, "Policy repository\nstatic rules (versioned)\npolicy-as-code", AMBER, fs=6.6)
    lp = box(ax, 2, 12, 22, 14, "Learned policy\npi(s) = argmax Q(s,a)\nfrom RL engine", GREEN, fs=6.6)
    cr = box(ax, 32, 22, 20, 16, "Conflict resolver\ndeny-overrides for\nlegal rules; learned\nchoice otherwise", PURPLE, fs=6.6)
    pdp = box(ax, 59, 22, 17, 16, "Policy decision\npoint (PDP)\nfinal action +\nexplanation", NAVY, fs=6.6)
    pep = box(ax, 82, 22, 16, 16, "Policy\nenforcement\npoint (PEP)", TEAL, fs=6.6)
    arrow(ax, (24, 41), (32, 33), c=AMBER); arrow(ax, (24, 19), (32, 27), c=GREEN)
    arrow(ax, (52, 30), (59, 30), c=GREY); arrow(ax, (76, 30), (82, 30), c=GREY)
    pa = box(ax, 34, 1, 44, 10, "Preference learning & continuous optimisation\n(reward feedback, drift alarms, DPO overrides)", GREEN, fs=6.4)
    arrow(ax, (90, 22), (74, 7), c=GREY, ls="--"); arrow(ax, (40, 7), (13, 12), c=GREEN, ls="--")
    save(fig, 11, "Dynamic Privacy Policy Framework")


# ---------------------------------------------------------------- Fig 12
def fig12():
    fig, ax = canvas(w=6.2, h=4.6, ymax=74, xmax=84)
    st = [("Draft\nbaseline rules", AMBER), ("Learn\nin simulation", GREEN), ("Validate\nshield + tests", PURPLE),
          ("Deploy\n(canary)", NAVY), ("Monitor\nKPIs & drift", TEAL), ("Refine\nretrain / roll back", RED)]
    cx, cy, R = 42, 36, 25
    for i, (t, c) in enumerate(st):
        th = np.deg2rad(90 - i * 60)
        x, y = cx + R * np.cos(th), cy + R * np.sin(th)
        box(ax, x - 9, y - 5.5, 18, 11, t, c, fs=6.8)
        th2 = np.deg2rad(90 - (i + 1) * 60)
        a1 = th - np.deg2rad(17); a2 = th2 + np.deg2rad(17)
        arrow(ax, (cx + R * np.cos(a1), cy + R * np.sin(a1)), (cx + R * np.cos(a2), cy + R * np.sin(a2)), c=GREY, rad=-0.25)
    ax.text(cx, cy, "Policy\nversion v_k", ha="center", va="center", fontsize=8.5, fontweight="bold", color=NAVY)
    save(fig, 12, "Privacy Policy Evolution Lifecycle")


# ---------------------------------------------------------------- Fig 13
def fig13():
    fig, ax = canvas(h=3.4, ymax=44)
    inp = [("Mechanism a\nleakage L(a)", TEAL), ("Attribute\nsensitivity w_s", AMBER), ("Threat level\nmultiplier m_th", RED), ("Requester\ntrust f_tr", NAVY)]
    for i, (t, c) in enumerate(inp):
        box(ax, 1, 34 - i * 10, 17, 8, t, c, fs=6.5)
        arrow(ax, (18, 38 - i * 10), (25, 23), c=c, lw=0.8)
    r = box(ax, 25, 15, 17, 16, "Risk score\nRisk = L.w.m.f.d\n(Eq. 8)", RED, fs=7, bold=True)
    b = box(ax, 48, 15, 15, 16, "Risk band\nlow < 0.15\nmedium < 0.40\nhigh >= 0.40", PURPLE, fs=6.6)
    arrow(ax, (42, 23), (48, 23), c=GREY)
    mit = [("Low: release with\nchosen mechanism", GREEN), ("Medium: stronger\nnoise / generalise", AMBER), ("High: deny, alert,\nrequire approval", RED)]
    for i, (t, c) in enumerate(mit):
        box(ax, 70, 32 - i * 12, 28, 9, t, c, fs=6.5)
        arrow(ax, (63, 23), (70, 36.5 - i * 12), c=c, lw=0.8)
    save(fig, 13, "Privacy Risk Assessment Workflow")


# ---------------------------------------------------------------- Fig 14
def fig14():
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    quad = [(0, 0, "#DDEFE0", "Low risk, low value\nDP e=0.1 or deny\n(cheap to refuse)"),
            (0.5, 0, "#D7F0EC", "Low risk, high value\nDP e=1.0 / k-anon\n(release)"),
            (0, 0.5, "#F6DCD6", "High risk, low value\nDENY\n(no justification)"),
            (0.5, 0.5, "#FBEBD0", "High risk, high value\nDP e=0.1-0.5 + approval\n(balance, explain)")]
    for x, y, c, t in quad:
        ax.add_patch(plt.Rectangle((x, y), 0.5, 0.5, color=c))
        ax.text(x + 0.25, y + 0.25, t, ha="center", va="center", fontsize=8)
    ax.set_xlabel("Expected utility of the request (purpose-weighted)")
    ax.set_ylabel("Assessed privacy risk")
    ax.set_title("Privacy decision intelligence: value-risk decision space", fontsize=9, color=NAVY)
    ax.axhline(0.5, color="white", lw=2); ax.axvline(0.5, color="white", lw=2)
    ax.annotate("learned boundary\nshifts with context", xy=(0.80, 0.58), xytext=(0.56, 0.86), fontsize=7,
                arrowprops=dict(arrowstyle="->", color=NAVY))
    xs = np.linspace(0, 1, 50); ax.plot(xs, 0.15 + 0.6 * xs ** 1.5, "--", color=NAVY, lw=1.2)
    save(fig, 14, "Privacy Decision Intelligence Framework")


# ---------------------------------------------------------------- Fig 15
def fig15():
    fig, ax = canvas(h=3.3, ymax=42)
    st = [("Aggregate query\nf(D) on cohort", GREY), ("Sensitivity\nanalysis\nDf = range/n", TEAL), ("Agent picks e\n{0.1, 0.5, 1.0}", GREEN),
          ("Laplace noise\nb = Df / e\n(Eq. 10)", NAVY), ("Accountant\ncharges e\n(Eq. 12)", AMBER), ("Noisy release\nf(D) + Lap(b)", PURPLE)]
    for i, (t, c) in enumerate(st):
        box(ax, 1 + i * 16.5, 16, 14, 14, t, c, fs=6.6)
        if i:
            arrow(ax, (1 + i * 16.5 - 2.5, 23), (1 + i * 16.5, 23), c=GREY)
    arrow(ax, (73, 16), (41, 16), c=AMBER, ls="--", rad=-0.45)
    ax.text(57, 4, "remaining budget feeds back into state s_t", ha="center", fontsize=6.5, color=AMBER)
    save(fig, 15, "Differential Privacy Management Framework")


# ---------------------------------------------------------------- Fig 17
def fig17():
    fig, ax = canvas(h=3.8, ymax=50)
    q = box(ax, 2, 18, 16, 14, "Learned\nQ(s, .) and\nshield mask", GREEN, fs=6.8)
    techs = [("Attribute influence\n(counterfactual flips)", PURPLE), ("Decision confidence\nC(s) (Eq. 13)", NAVY),
             ("Rule / constraint\ntrace (shield)", AMBER), ("Counterfactual\n'what would change'", TEAL)]
    for i, (t, c) in enumerate(techs):
        box(ax, 26, 39 - i * 11, 22, 8.5, t, c, fs=6.4)
        arrow(ax, (18, 25), (26, 43 - i * 11), c=GREY, lw=0.7)
        arrow(ax, (48, 43 - i * 11), (56, 25), c=GREY, lw=0.7)
    nl = box(ax, 56, 16, 18, 18, "Explanation\ncomposer\n(templates, no\nfree generation)", NAVY, fs=6.6)
    aud = [("Data subject\nplain language", TEAL), ("Requester\nwhy denied / noised", GREEN), ("DPO / regulator\naudit record", AMBER)]
    for i, (t, c) in enumerate(aud):
        box(ax, 80, 36 - i * 13, 18, 10, t, c, fs=6.4)
        arrow(ax, (74, 25), (80, 41 - i * 13), c=c, lw=0.8)
    save(fig, 17, "Explainable Privacy Intelligence Framework")


# ---------------------------------------------------------------- Fig 19
def fig19():
    fig, ax = canvas(h=4.0, ymax=54)
    regs = ["GDPR (EU)", "HIPAA (US)", "DPDP Act 2023 (IN)", "ISO/IEC 27701"]
    for i, r in enumerate(regs):
        box(ax, 1, 43 - i * 10, 16, 7.5, r, AMBER, fs=6.8)
        arrow(ax, (17, 46.7 - i * 10), (23, 31), c=AMBER, lw=0.7)
    pc = box(ax, 23, 24, 17, 14, "Policy-as-code\nmachine-readable\nobligations", AMBER, fs=6.6)
    sh = box(ax, 46, 24, 17, 14, "Compliance oracle\n& action shield\n(pre-decision)", RED, fs=6.6)
    au = box(ax, 69, 33, 14, 12, "Immutable\naudit log", NAVY, fs=6.6)
    mo = box(ax, 69, 14, 14, 12, "Continuous\ncompliance\nmonitor", TEAL, fs=6.6)
    dpo = box(ax, 87, 24, 12, 14, "DPO review\n& regulator\nreports", PURPLE, fs=6.4)
    arrow(ax, (40, 31), (46, 31), c=GREY); arrow(ax, (63, 33), (69, 39), c=GREY); arrow(ax, (63, 28), (69, 20), c=GREY)
    arrow(ax, (83, 39), (87, 33), c=GREY); arrow(ax, (83, 20), (87, 28), c=GREY)
    arrow(ax, (93, 24), (31, 24), c=PURPLE, ls="--", rad=-0.45)
    ax.text(62, 2.5, "rule updates and DPO overrides", ha="center", fontsize=6.5, color=PURPLE)
    save(fig, 19, "Privacy Governance Architecture")


# ---------------------------------------------------------------- Fig 20
def fig20():
    fig, ax = canvas(h=3.8, ymax=50)
    src = [("Access & query logs", GREY), ("Network / host IDS", GREY), ("External threat intel", GREY), ("Privacy-attack probes\n(membership, linkage)", RED)]
    for i, (t, c) in enumerate(src):
        box(ax, 1, 40 - i * 10, 18, 8, t, c, fs=6.3)
        arrow(ax, (19, 44 - i * 10), (25, 27), c=GREY, lw=0.7)
    ad = box(ax, 25, 20, 16, 14, "Anomaly detection\nquery-rate, overlap,\ndifferencing", RED, fs=6.5)
    tl = box(ax, 47, 20, 15, 14, "Threat level\nlow / elevated /\nhigh (Markov)", AMBER, fs=6.5)
    arrow(ax, (41, 27), (47, 27), c=GREY)
    ag = box(ax, 68, 31, 15, 11, "RL agent state\n(threat feature)", GREEN, fs=6.5)
    rs = box(ax, 68, 12, 15, 11, "Automated defence\nrate-limit, deny,\nre-key, alert SOC", NAVY, fs=6.3)
    arrow(ax, (62, 29), (68, 36), c=GREY); arrow(ax, (62, 25), (68, 18), c=GREY)
    fb = box(ax, 87, 20, 12, 14, "SOC\nanalyst\nfeedback", PURPLE, fs=6.4)
    arrow(ax, (83, 17), (87, 25), c=GREY); arrow(ax, (87, 29), (83, 36), c=GREY)
    save(fig, 20, "Privacy Threat Intelligence Framework")


# ---------------------------------------------------------------- Fig 21
def fig21():
    fig, ax = canvas(h=3.6, ymax=48)
    d = box(ax, 1, 28, 18, 14, "WDBC health\ndataset\n569 x 30 (CSV)", GREY, fs=6.6)
    qp = box(ax, 1, 6, 18, 14, "Query pool\n4,000 cohort-mean\nqueries", TEAL, fs=6.6)
    arrow(ax, (10, 28), (10, 20), c=GREY)
    env = box(ax, 25, 14, 22, 20, "PrivacyEnv (Python)\ncontext generator,\nmechanisms, risk,\ncompliance oracle,\nconcept drift @ 30k", NAVY, fs=6.4)
    arrow(ax, (19, 13), (25, 20), c=GREY); arrow(ax, (19, 35), (25, 28), c=GREY)
    ags = ["Static-Strict", "Static-Permissive", "Rule-based", "Contextual bandit", "Q-learning", "SAPM-RL (proposed)"]
    for i, t in enumerate(ags):
        box(ax, 54, 41 - i * 7, 18, 5.6, t, GREEN if i == 5 else PURPLE, fs=6.3)
        arrow(ax, (47, 24), (54, 43.8 - i * 7), c="#9CA3AF", lw=0.6, style="<|-|>")
    lg = box(ax, 78, 26, 20, 12, "Step logger\nreward, utility,\nrisk, violation", AMBER, fs=6.4)
    mt = box(ax, 78, 8, 20, 12, "Metrics & plots\n5 seeds x 60k steps\nmean +/- s.d.", TEAL, fs=6.4)
    arrow(ax, (72, 30), (78, 32), c=GREY); arrow(ax, (88, 26), (88, 20), c=GREY)
    save(fig, 21, "Experimental Testbed Architecture")


# ---------------------------------------------------------------- Fig 22
def fig22():
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.axis("equal"); ax.axis("off")
    fams = [("Privacy\neffectiveness", RED, "risk, loss/request"), ("Utility\npreservation", TEAL, "U = exp(-RE/tau)"),
            ("Compliance\nreadiness", AMBER, "violation rate"), ("Adaptation", GREEN, "post-drift reward,\nadaptation time"),
            ("RL\nconvergence", PURPLE, "rolling reward"), ("Trust &\nfairness", NAVY, "group parity,\nrobustness")]
    for i, (t, c, d) in enumerate(fams):
        ax.add_patch(Wedge((0, 0), 1.0, i * 60, (i + 1) * 60, width=0.45, facecolor=LIGHT[c], edgecolor="white", lw=2))
        th = np.deg2rad(i * 60 + 30)
        ax.text(0.72 * np.cos(th), 0.72 * np.sin(th), t, ha="center", va="center", fontsize=6.3, fontweight="bold", color=c)
        ax.text(1.28 * np.cos(th), 1.28 * np.sin(th), d, ha="center", va="center", fontsize=7, color="#374151")
    ax.add_patch(Circle((0, 0), 0.5, color=NAVY))
    ax.text(0, 0, "Composite\nreward\n(Eq. 3)", ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)
    save(fig, 22, "Evaluation Framework")


# ---------------------------------------------------------------- Fig 27
def fig27():
    fig, ax = canvas(h=3.8, ymax=50)
    ax.add_patch(plt.Polygon([[4, 40], [50, 49], [96, 40]], color=NAVY))
    ax.text(50, 43, "Trustworthy self-adaptive privacy", ha="center", color="white", fontsize=8.5, fontweight="bold")
    pil = [("Robustness\nnoisy context,\ndrift alarms", RED), ("Fairness\nparity across\ntrust & purpose", TEAL), ("Accountability\naudit log,\nDPO override", AMBER),
           ("Transparency\nreasons &\nconfidence", PURPLE), ("Safety\ncompliance\nshield", GREEN), ("Resilience\nroll-back,\nsafe default", NAVY)]
    for i, (t, c) in enumerate(pil):
        box(ax, 5 + i * 15.3, 9, 13, 29, t, c, fs=6.6)
    ax.add_patch(plt.Rectangle((4, 3), 92, 5, color=GREY))
    ax.text(50, 5.5, "Foundation: privacy by design, data minimisation, measured evaluation", ha="center", color="white", fontsize=7)
    save(fig, 27, "Trustworthy Privacy Framework")


# ---------------------------------------------------------------- Fig 28
def fig28():
    fig, ax = canvas(h=4.2, ymax=56)
    core = box(ax, 36, 22, 28, 14, "Hospital SAPM-RL\nprivacy gateway", GREEN, fs=8, bold=True)
    nodes = [("EHR & lab systems", 2, 42, GREY), ("Wearables / IoMT", 2, 22, GREY), ("Patient consent app", 2, 2, TEAL),
             ("Treating clinicians\n(raw, high trust)", 72, 42, NAVY), ("Research cohort\nstudies (DP)", 72, 22, PURPLE),
             ("Insurers / marketing\n(deny by default)", 72, 2, RED)]
    for t, x, y, c in nodes:
        box(ax, x, y, 26, 10, t, c, fs=6.8)
        if x < 50:
            arrow(ax, (28, y + 5), (36, 29), c=c, lw=0.8)
        else:
            arrow(ax, (64, 29), (72, y + 5), c=c, lw=0.8)
    ax.text(50, 12, "HIPAA minimum-necessary + DPDP consent\nchecked by the compliance shield", ha="center", fontsize=6.8, color=AMBER)
    save(fig, 28, "Healthcare Privacy Intelligence Ecosystem")


# ---------------------------------------------------------------- Fig 29
def fig29():
    fig, ax = canvas(h=4.0, ymax=54)
    ed = [("Traffic cameras", 2), ("Smart meters", 18), ("Mobility / transit", 34), ("Public Wi-Fi", 50), ("Environmental IoT", 66), ("Citizen services", 82)]
    for t, x in ed:
        box(ax, x, 42, 15, 7, t, GREY, fs=6.4)
        arrow(ax, (x + 7.5, 42), (x + 7.5, 36), c=GREY, lw=0.7)
    header(ax, 2, 29, 95, 6, "Edge privacy agents: local context, on-device aggregation, per-ward privacy budget", TEAL, fs=7)
    box(ax, 2, 14, 46, 11, "City data platform: SAPM-RL\nmulti-agent coordination, shared accountant", GREEN, fs=6.8)
    box(ax, 51, 14, 46, 11, "Governance: municipal DPO, open-data\nportal (DP releases), citizen dashboard", AMBER, fs=6.8)
    arrow(ax, (25, 29), (25, 25), c=TEAL); arrow(ax, (74, 29), (74, 25), c=TEAL)
    box(ax, 10, 2, 80, 8, "Consumers: traffic control, energy planning, emergency services, researchers", NAVY, fs=6.6)
    arrow(ax, (25, 14), (35, 10), c=GREY); arrow(ax, (74, 14), (65, 10), c=GREY)
    save(fig, 29, "Smart City Adaptive Privacy Framework")


# ---------------------------------------------------------------- Fig 31
def fig31():
    fig, ax = canvas(h=3.8, ymax=50)
    rows = [("Reward engineering", "weights encode values; mis-set\nlambda shifts the trade-off", "constraint form, stakeholder\nweight elicitation", RED),
            ("Adaptation instability", "exploration after drift can\nbriefly lower utility", "shield + safe defaults,\ncanary deployment", AMBER),
            ("Scalability", "tabular Q grows as 3^k\nwith attributes", "function approximation,\nfactored / deep RL", NAVY),
            ("Computational overhead", "per-request inference,\naccounting, logging", "caching, edge agents,\nbatched updates", TEAL),
            ("Regulatory uncertainty", "rules change faster\nthan code", "policy-as-code, legal\nreview loop", PURPLE)]
    header(ax, 2, 44, 24, 4.5, "Challenge", NAVY, fs=7); header(ax, 28, 44, 34, 4.5, "Limitation observed", RED, fs=7); header(ax, 64, 44, 34, 4.5, "Mitigation path", GREEN, fs=7)
    for i, (a, b, c, col) in enumerate(rows):
        y = 35 - i * 8.3
        box(ax, 2, y, 24, 7, a, col, fs=6.6, bold=True)
        box(ax, 28, y, 34, 7, b, RED, fs=6.2, fill="white")
        box(ax, 64, y, 34, 7, c, GREEN, fs=6.2, fill="white")
        arrow(ax, (62, y + 3.5), (64, y + 3.5), c=GREY)
    save(fig, 31, "Challenges and Limitations Roadmap")


# ---------------------------------------------------------------- Fig 32
def fig32():
    fig, ax = canvas(h=3.8, ymax=50)
    hz = [("Near term (1-2 yrs)", ["Deep / factored RL\nfor large contexts", "User studies on\nexplanations", "Real SIEM & consent\nintegration"], TEAL),
          ("Mid term (2-4 yrs)", ["Multi-agent RL across\nfederated sites", "Privacy-aware\nfoundation models", "Formal verification\nof shields"], NAVY),
          ("Long term (4+ yrs)", ["Cognitive privacy\nassistants", "Quantum-safe\nprivacy architectures", "Autonomous,\ncertifiable agents"], PURPLE)]
    for i, (t, items, c) in enumerate(hz):
        x = 2 + i * 33
        header(ax, x, 40, 30, 6, t, c, fs=7.5)
        for j, it in enumerate(items):
            box(ax, x + 1, 29 - j * 11, 28, 9, it, c, fs=6.6)
        if i < 2:
            arrow(ax, (x + 30, 43), (x + 33, 43), c=GREY)
    save(fig, 32, "Future Research Roadmap")


# ---------------------------------------------------------------- Fig 33
def fig33():
    fig, ax = canvas(h=4.2, ymax=56)
    ax.add_patch(Circle((50, 28), 22, facecolor=LIGHT[NAVY], edgecolor=NAVY))
    ax.add_patch(Circle((50, 28), 13, facecolor=LIGHT[GREEN], edgecolor=GREEN))
    ax.text(50, 28, "Autonomous\nprivacy agents\n(MARL)", ha="center", va="center", fontsize=8, fontweight="bold", color=GREEN)
    ring = ["Privacy-aware\nLLMs", "Federated\nlearning", "Post-quantum\ncrypto", "Digital twins\nfor policy test", "Personal data\nstores", "Regulatory\nsandboxes"]
    for i, t in enumerate(ring):
        th = np.deg2rad(90 + i * 60)
        ax.text(50 + 17.5 * np.cos(th) * 1.0, 28 + 17.5 * np.sin(th) * 0.95, t, ha="center", va="center", fontsize=6.4, color=NAVY)
    outer = [("Citizens", 6, 46), ("Enterprises", 80, 46), ("Regulators", 6, 6), ("Researchers", 80, 6)]
    for t, x, y in outer:
        box(ax, x, y, 14, 6, t, AMBER, fs=6.8)
    save(fig, 33, "Next-Generation Adaptive Privacy Ecosystem")


# ---------------------------------------------------------------- Fig 34
def fig34():
    fig, ax = canvas(h=3.8, ymax=50)
    cs = [("C1 Architecture", "MAPE-K style SAPM-RL\nwith 8 layers", NAVY), ("C2 Formulation", "context MDP, composite\nreward, risk model", GREEN),
          ("C3 Safe learning", "compliance shield +\nLagrangian + drift test", AMBER), ("C4 Evidence", "real-data testbed,\n6 methods, 5 seeds", TEAL),
          ("C5 Explainability", "attribute influence,\nconfidence score", PURPLE), ("C6 Practice", "use cases, governance\nmapping, roadmap", RED)]
    for i, (t, d, c) in enumerate(cs):
        x = 2 + (i % 3) * 33; y = 27 if i < 3 else 4
        header(ax, x, y + 12, 30, 5, t, c, fs=7.2)
        box(ax, x, y, 30, 11.5, d, c, fs=6.8)
    ax.text(50, 47, "Research contribution framework", ha="center", fontsize=9, fontweight="bold", color=NAVY)
    save(fig, 34, "Research Contribution Framework")


if __name__ == "__main__":
    for f in [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10, fig11, fig12, fig13, fig14, fig15,
              fig17, fig19, fig20, fig21, fig22, fig27, fig28, fig29, fig31, fig32, fig33, fig34]:
        f()
    print("concept figures done")
