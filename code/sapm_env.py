"""
SAPM-RL: Self-Adaptive Privacy Management using Reinforcement Learning
Environment module.

A stream of data-access requests arrives against a real, sensitive health
dataset (Wisconsin Diagnostic Breast Cancer, 569 records x 30 attributes,
bundled with scikit-learn). For every request the privacy agent observes a
discrete privacy context and chooses a privacy-protection action. The
environment answers the request with that mechanism on the real data, so the
utility term is measured from the actual error of the released answer.

Context (state) attributes, each with 3 levels:
    sensitivity  : low / medium / high     (sensitivity class of the attribute)
    trust        : low / medium / high     (requester trust / role)
    threat       : low / elevated / high   (security-event level, Markov chain)
    purpose      : research / clinical / marketing
    consent      : open / moderate / strict (data-subject preference)
    budget       : ample / limited / exhausted (privacy-budget accountant)

Actions:
    0 DENY        refuse the request
    1 DP_0.1      Laplace mechanism, epsilon = 0.1
    2 DP_0.5      Laplace mechanism, epsilon = 0.5
    3 DP_1.0      Laplace mechanism, epsilon = 1.0
    4 K_ANON      release computed on k-anonymised (generalised) records
    5 RAW         release computed on pseudonymised raw records
"""
import numpy as np
from sklearn.datasets import load_breast_cancer

ACTIONS = ["DENY", "DP_0.1", "DP_0.5", "DP_1.0", "K_ANON", "RAW"]
N_ACTIONS = len(ACTIONS)
EPS = {1: 0.1, 2: 0.5, 3: 1.0}
# privacy-loss units charged to the budget accountant per action
LOSS_UNITS = np.array([0.0, 0.1, 0.5, 1.0, 2.0, 4.0])
# leakage factor L(a) used in the risk function (Eq. 8)
LEAK = np.array([0.0, 1 - np.exp(-0.1), 1 - np.exp(-0.5), 1 - np.exp(-1.0), 0.45, 1.0])
ATTRS = ["sensitivity", "trust", "threat", "purpose", "consent", "budget"]
LEVELS = {
    "sensitivity": ["low", "medium", "high"],
    "trust": ["low", "medium", "high"],
    "threat": ["low", "elevated", "high"],
    "purpose": ["research", "clinical", "marketing"],
    "consent": ["open", "moderate", "strict"],
    "budget": ["ample", "limited", "exhausted"],
}
N_STATES = 3 ** 6


def encode(ctx):
    s = 0
    for v in ctx:
        s = s * 3 + int(v)
    return s


def decode(s):
    out = []
    for _ in range(6):
        out.append(s % 3)
        s //= 3
    return out[::-1]


class QueryPool:
    """Pre-built pool of aggregate queries on the real dataset."""

    def __init__(self, n_queries=4000, seed=0, k_bins=5):
        data = load_breast_cancer()
        X = data.data
        self.X = X
        self.feature_names = list(data.feature_names)
        rng = np.random.default_rng(seed)
        nrec, nfeat = X.shape
        # sensitivity class of each attribute: "worst *" features (most
        # diagnostic) are high, "mean *" medium, "* error" low
        self.feat_sens = np.array(
            [2 if n.startswith("worst") else (0 if n.endswith("error") else 1) for n in self.feature_names]
        )
        # generalised copy of the data (quantile bins -> bin midpoints)
        Xg = np.empty_like(X)
        for j in range(nfeat):
            qs = np.quantile(X[:, j], np.linspace(0, 1, k_bins + 1))
            idx = np.clip(np.searchsorted(qs, X[:, j], side="right") - 1, 0, k_bins - 1)
            mids = (qs[:-1] + qs[1:]) / 2
            Xg[:, j] = mids[idx]
        self.Xg = Xg
        self.f = rng.integers(0, nfeat, n_queries)
        self.n = rng.integers(40, 300, n_queries)
        self.true = np.empty(n_queries)
        self.gen = np.empty(n_queries)
        self.rng_width = X.max(0) - X.min(0)
        for q in range(n_queries):
            rows = rng.choice(nrec, self.n[q], replace=False)
            self.true[q] = X[rows, self.f[q]].mean()
            self.gen[q] = Xg[rows, self.f[q]].mean()
        self.size = n_queries
        # expected utility of each action per query (Monte-Carlo, for oracle)
        self.exp_util = np.zeros((n_queries, N_ACTIONS))
        for q in range(n_queries):
            for a in range(N_ACTIONS):
                self.exp_util[q, a] = np.mean([self.utility(q, a, rng) for _ in range(40)])

    def release(self, q, a, rng):
        """Answer query q with action a. Returns released value (None if denied)."""
        if a == 0:
            return None
        if a in EPS:
            scale = (self.rng_width[self.f[q]] / self.n[q]) / EPS[a]
            return self.true[q] + rng.laplace(0, scale)
        if a == 4:
            return self.gen[q]
        return self.true[q]

    def utility(self, q, a, rng, tau=0.05):
        """Eq. (11): U = exp(-relative_error / tau); 0 if denied."""
        v = self.release(q, a, rng)
        if v is None:
            return 0.0
        rel = abs(v - self.true[q]) / (abs(self.true[q]) + 1e-9)
        return float(np.exp(-rel / tau))


class PrivacyEnv:
    """Non-stationary privacy-management environment with a concept drift."""

    SENS_W = np.array([0.3, 0.6, 1.0])
    THREAT_M = np.array([1.0, 1.5, 2.5])
    TRUST_F = np.array([1.2, 1.0, 0.7])
    PURPOSE_W = np.array([1.0, 1.3, 0.5])

    def __init__(self, pool, seed=0, drift_at=30000, lam=0.6, kappa=2.0,
                 window=1000, budget=1200.0, obs_noise=0.0):
        self.pool = pool
        self.rng = np.random.default_rng(seed)
        self.drift_at = drift_at
        self.lam, self.kappa = lam, kappa
        self.window, self.budget = window, budget
        self.obs_noise = obs_noise
        self.t = 0
        self.threat = 0
        self.spent = 0.0
        self._new_request()

    # --- regime parameters (change at the drift point) ---
    def _regime(self):
        post = self.t >= self.drift_at
        if not post:
            P = np.array([[0.90, 0.08, 0.02], [0.30, 0.60, 0.10], [0.10, 0.30, 0.60]])
            consent_p = [0.45, 0.40, 0.15]
            purpose_p = [0.45, 0.40, 0.15]
        else:  # new threat campaign + stricter user sentiment after a breach
            P = np.array([[0.70, 0.20, 0.10], [0.15, 0.55, 0.30], [0.05, 0.20, 0.75]])
            consent_p = [0.20, 0.40, 0.40]
            purpose_p = [0.35, 0.40, 0.25]
        return P, consent_p, purpose_p

    def _new_request(self):
        _, consent_p, purpose_p = self._regime()
        self.q = int(self.rng.integers(self.pool.size))
        self.sens = int(self.pool.feat_sens[self.pool.f[self.q]])
        self.trust = int(self.rng.choice(3, p=[0.25, 0.45, 0.30]))
        self.purpose = int(self.rng.choice(3, p=purpose_p))
        self.consent = int(self.rng.choice(3, p=consent_p))

    def budget_level(self):
        frac = self.spent / self.budget
        return 0 if frac < 0.6 else (1 if frac < 1.0 else 2)

    def context(self):
        return [self.sens, self.trust, self.threat, self.purpose, self.consent, self.budget_level()]

    def observe(self):
        ctx = self.context()
        if self.obs_noise > 0 and self.rng.random() < self.obs_noise:
            ctx[2] = int(self.rng.integers(3))  # mis-reported threat level
        return encode(ctx)

    # --- compliance oracle (GDPR/DPDP-style purpose limitation + consent) ---
    def violation(self, a, ctx=None):
        sens, trust, threat, purpose, consent, budget = ctx if ctx is not None else self.context()
        if a == 0:
            return 0
        if purpose == 2 and consent != 0:
            return 1  # marketing without open consent
        if consent == 2 and a >= 4:
            return 1  # strict consent: record-level release forbidden
        if sens == 2 and a == 5 and not (purpose == 1 and trust == 2):
            return 1  # raw high-sensitivity data only for trusted clinicians
        if budget == 2:
            return 1  # privacy budget exhausted
        return 0

    def risk(self, a, ctx=None):
        sens, trust, threat, purpose, consent, budget = ctx if ctx is not None else self.context()
        post = 1.25 if self.t >= self.drift_at else 1.0
        return LEAK[a] * self.SENS_W[sens] * self.THREAT_M[threat] * self.TRUST_F[trust] * post

    def expected_reward(self, a):
        ctx = self.context()
        u = self.pool.exp_util[self.q, a] * self.PURPOSE_W[self.purpose]
        return u - self.lam * self.risk(a, ctx) - self.kappa * self.violation(a, ctx)

    def oracle_action(self):
        return int(np.argmax([self.expected_reward(a) for a in range(N_ACTIONS)]))

    def step(self, a):
        ctx = self.context()
        u_raw = self.pool.utility(self.q, a, self.rng)
        u = u_raw * self.PURPOSE_W[self.purpose]
        rk = self.risk(a, ctx)
        v = self.violation(a, ctx)
        r = u - self.lam * rk - self.kappa * v                      # Eq. (3)
        info = dict(utility=u_raw, risk=rk, violation=v, loss=LOSS_UNITS[a],
                    sens=ctx[0], trust=ctx[1], threat=ctx[2], purpose=ctx[3], consent=ctx[4])
        # accountant
        self.spent += LOSS_UNITS[a] if v == 0 else 0.0
        self.t += 1
        if self.t % self.window == 0:
            self.spent = 0.0
        # threat dynamics: record-level releases during elevated threat raise
        # the chance of an incident (keeps threat high) -> actions shape the future
        P, _, _ = self._regime()
        p = P[self.threat].copy()
        if a >= 4 and self.threat >= 1:
            p = 0.5 * p + 0.5 * np.array([0.0, 0.3, 0.7])
        self.threat = int(self.rng.choice(3, p=p / p.sum()))
        self._new_request()
        return self.observe(), r, info
