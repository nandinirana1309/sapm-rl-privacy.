"""
Runs every experiment reported in the case study and writes results to ../results.
Usage:  python run_experiments.py            (about 5-10 minutes on a laptop)
"""
import json, os, time
import numpy as np
import pandas as pd
from sapm_env import QueryPool, PrivacyEnv, N_ACTIONS, ACTIONS, ATTRS, decode, encode
from sapm_agents import StaticAgent, RuleBasedAgent, QAgent, SAPMAgent

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUT, exist_ok=True); os.makedirs(DATA, exist_ok=True)
STEPS, DRIFT, SEEDS = 60000, 30000, [0, 1, 2, 3, 4]


def make_agents(seed):
    return [
        StaticAgent(1, "Static-Strict (DP e=0.1)"),
        StaticAgent(4, "Static-Permissive (k-anon)"),
        RuleBasedAgent(),
        QAgent("Contextual bandit", gamma=0.0, seed=seed),
        QAgent("Q-learning", gamma=0.6, seed=seed),
        SAPMAgent(gamma=0.6, seed=seed),
    ]


def run(agent, pool, seed, steps=STEPS, drift=DRIFT, lam=0.6, obs_noise=0.0, log_all=True):
    env = PrivacyEnv(pool, seed=seed + 100, drift_at=drift, lam=lam, obs_noise=obs_noise)
    s = env.observe()
    keys = ["reward", "utility", "risk", "violation", "loss", "oracle_match", "action",
            "trust", "purpose", "threat"]
    log = {k: np.zeros(steps) for k in keys}
    for t in range(steps):
        oa = env.oracle_action()
        a = agent.act(s, env)
        s2, r, info = env.step(a)
        agent.update(s, a, r, s2, info)
        log["reward"][t] = r; log["utility"][t] = info["utility"]; log["risk"][t] = info["risk"]
        log["violation"][t] = info["violation"]; log["loss"][t] = info["loss"]
        log["oracle_match"][t] = float(a == oa); log["action"][t] = a
        log["trust"][t] = info["trust"]; log["purpose"][t] = info["purpose"]; log["threat"][t] = info["threat"]
        s = s2
    return log


def roll(x, w=1000):
    c = np.cumsum(np.insert(x, 0, 0))
    return (c[w:] - c[:-w]) / w


def adaptation_time(rew, drift=DRIFT, w=1000, hold=3000):
    """Steps after the drift until the rolling reward reaches the post-drift
    steady level (within 0.02) and stays there for `hold` steps (Eq. 15)."""
    r = roll(rew, w)
    steady = rew[-10000:].mean()
    pre = rew[drift - 10000:drift].mean()
    post_curve = r[drift - w + 1:]
    ok = post_curve >= steady - 0.02
    for i in range(len(ok) - hold):
        if ok[i:i + hold].all():
            return int(i), pre, steady
    return -1, pre, steady


def main():
    t0 = time.time()
    pool = QueryPool(n_queries=4000, seed=7)
    # ---- datasets folder ----
    df = pd.DataFrame(pool.X, columns=pool.feature_names)
    df.to_csv(os.path.join(DATA, "wdbc_health_records.csv"), index=False)
    pd.DataFrame({"query_id": range(pool.size),
                  "feature": [pool.feature_names[i] for i in pool.f],
                  "sensitivity_class": [["low", "medium", "high"][pool.feat_sens[i]] for i in pool.f],
                  "cohort_size": pool.n, "true_mean": pool.true, "generalised_mean": pool.gen
                  }).to_csv(os.path.join(DATA, "query_pool.csv"), index=False)

    summary, curves = [], {}
    per_group = []
    for seed in SEEDS:
        for ag in make_agents(seed):
            log = run(ag, pool, seed)
            post = slice(DRIFT, STEPS); pre = slice(DRIFT - 10000, DRIFT)
            at, pre_r, steady = adaptation_time(log["reward"])
            row = dict(agent=ag.name, seed=seed,
                       reward=log["reward"].mean(), utility=log["utility"].mean(),
                       risk=log["risk"].mean(), violation=log["violation"].mean(),
                       loss_per_req=log["loss"].mean(),
                       oracle_match=log["oracle_match"].mean(),
                       reward_pre=log["reward"][pre].mean(), reward_post=log["reward"][post].mean(),
                       viol_post=log["violation"][post].mean(),
                       final_reward=log["reward"][-10000:].mean(),
                       final_utility=log["utility"][-10000:].mean(),
                       final_risk=log["risk"][-10000:].mean(),
                       final_violation=log["violation"][-10000:].mean(),
                       final_oracle=log["oracle_match"][-10000:].mean(),
                       adapt_steps=at,
                       drift_alarms=len(getattr(ag, "drift_alarms", [])),
                       first_alarm=(min([x for x in getattr(ag, "drift_alarms", []) if x > DRIFT], default=-1)))
            summary.append(row)
            curves.setdefault(ag.name, []).append(roll(log["reward"], 1000)[::100])
            curves.setdefault(ag.name + "|viol", []).append(roll(log["violation"], 1000)[::100])
            curves.setdefault(ag.name + "|oracle", []).append(roll(log["oracle_match"], 1000)[::100])
            # fairness: utility & denial by trust and purpose (final 20k steps)
            fin = slice(STEPS - 20000, STEPS)
            for g, name in [("trust", ["low", "medium", "high"]), ("purpose", ["research", "clinical", "marketing"])]:
                for lv in range(3):
                    m = log[g][fin] == lv
                    per_group.append(dict(agent=ag.name, seed=seed, attribute=g, level=name[lv],
                                          utility=log["utility"][fin][m].mean(),
                                          deny=(log["action"][fin][m] == 0).mean(),
                                          violation=log["violation"][fin][m].mean()))
            if seed == 0:
                np.save(os.path.join(OUT, f"actions_{ag.name.split()[0]}.npy"), log["action"])
                if isinstance(ag, SAPMAgent):
                    np.save(os.path.join(OUT, "sapm_Q.npy"), ag.Q)
                    json.dump(dict(alarms=ag.drift_alarms, mu=ag.mu), open(os.path.join(OUT, "sapm_meta.json"), "w"))
            print(f"seed {seed} {ag.name:28s} R={row['reward']:.3f} U={row['utility']:.3f} "
                  f"risk={row['risk']:.3f} viol={row['violation']:.3f} adapt={at} ({time.time()-t0:.0f}s)", flush=True)

    S = pd.DataFrame(summary); S.to_csv(os.path.join(OUT, "summary_all_seeds.csv"), index=False)
    agg = S.groupby("agent", sort=False).agg(["mean", "std"]); agg.to_csv(os.path.join(OUT, "summary_mean_std.csv"))
    pd.DataFrame(per_group).groupby(["agent", "attribute", "level"], sort=False).mean(numeric_only=True)\
        .drop(columns="seed").to_csv(os.path.join(OUT, "fairness_groups.csv"))
    np.savez(os.path.join(OUT, "curves.npz"), **{k: np.array(v) for k, v in curves.items()})

    # ---- privacy-utility trade-off: lambda sweep for SAPM + fixed mechanisms ----
    trade = []
    for lam in [0.1, 0.3, 0.6, 0.8, 1.2, 1.8, 2.5, 4.0]:
        for seed in [0, 1]:
            ag = SAPMAgent(gamma=0.6, seed=seed, risk_target=10.0)  # constraint off: pure lambda sweep
            log = run(ag, pool, seed, steps=40000, drift=10**9, lam=lam)
            fin = slice(30000, 40000)
            trade.append(dict(method="SAPM-RL", param=lam, utility=log["utility"][fin].mean(),
                              risk=log["risk"][fin].mean(), violation=log["violation"][fin].mean()))
    for a, nm in [(1, "DP e=0.1"), (2, "DP e=0.5"), (3, "DP e=1.0"), (4, "k-anon"), (5, "Raw")]:
        log = run(StaticAgent(a, nm), pool, 0, steps=20000, drift=10**9)
        trade.append(dict(method="Static", param=nm, utility=log["utility"].mean(),
                          risk=log["risk"].mean(), violation=log["violation"].mean()))
    pd.DataFrame(trade).to_csv(os.path.join(OUT, "tradeoff.csv"), index=False)

    # ---- robustness: trained SAPM policy under observation noise ----
    rob = []
    for noise in [0.0, 0.1, 0.2, 0.3, 0.4]:
        for seed in [0, 1, 2]:
            for ag in [SAPMAgent(gamma=0.6, seed=seed), QAgent("Q-learning", seed=seed), RuleBasedAgent()]:
                log = run(ag, pool, seed, steps=40000, drift=10**9, obs_noise=noise)
                fin = slice(30000, 40000)
                rob.append(dict(agent=ag.name, noise=noise, reward=log["reward"][fin].mean(),
                                violation=log["violation"][fin].mean(), risk=log["risk"][fin].mean()))
    pd.DataFrame(rob).groupby(["agent", "noise"], sort=False).mean(numeric_only=True)\
        .to_csv(os.path.join(OUT, "robustness.csv"))

    # ---- explainability on the learned SAPM Q-table ----
    Q = np.load(os.path.join(OUT, "sapm_Q.npy"))
    visited = np.flatnonzero(np.abs(Q).sum(1) > 0)
    greedy = Q.argmax(1)
    imp = {}
    rng = np.random.default_rng(0)
    for i, at in enumerate(ATTRS):
        changes = 0; total = 0
        for s in visited:
            ctx = decode(s)
            for v in range(3):
                if v == ctx[i]:
                    continue
                c2 = list(ctx); c2[i] = v
                s2 = encode(c2)
                if np.abs(Q[s2]).sum() == 0:
                    continue
                total += 1; changes += greedy[s2] != greedy[s]
        imp[at] = changes / max(total, 1)
    srt = np.sort(Q[visited], axis=1)
    margin = srt[:, -1] - srt[:, -2]
    conf = 1 / (1 + np.exp(-margin / 0.1))  # explainability confidence (Eq. 13)
    json.dump(dict(importance=imp, conf_mean=float(conf.mean()), conf_median=float(np.median(conf)),
                   conf_hist=np.histogram(conf, bins=10, range=(0.5, 1))[0].tolist(),
                   visited=int(len(visited)),
                   action_share={ACTIONS[a]: float((greedy[visited] == a).mean()) for a in range(N_ACTIONS)}),
              open(os.path.join(OUT, "explainability.json"), "w"), indent=2)
    print("done in", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
