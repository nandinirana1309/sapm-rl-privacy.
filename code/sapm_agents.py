"""
Privacy-management agents: static, rule-based, bandit, Q-learning and the
proposed SAPM-RL agent (shielded, drift-aware, constraint-adaptive Q-learning).
"""
import numpy as np
from sapm_env import N_ACTIONS, N_STATES, decode, LOSS_UNITS


class StaticAgent:
    """Fixed privacy configuration regardless of context."""
    def __init__(self, action, name):
        self.a, self.name = action, name
    def act(self, s, env=None):
        return self.a
    def update(self, *args):
        pass


class RuleBasedAgent:
    """Expert ABAC-style policy written from pre-deployment knowledge."""
    name = "Rule-based"
    def act(self, s, env=None):
        sens, trust, threat, purpose, consent, budget = decode(s)
        if budget == 2 or purpose == 2:
            return 0
        if consent == 2:
            return 1
        if sens == 0 and trust == 2 and purpose == 1 and threat == 0:
            return 5
        if sens == 2:
            return 1 if threat == 2 else 2
        if threat == 2:
            return 2
        return 3
    def update(self, *args):
        pass


class QAgent:
    """Tabular Q-learning (gamma>0) or contextual bandit (gamma=0)."""
    def __init__(self, name="Q-learning", gamma=0.6, alpha=0.1, eps0=1.0,
                 eps_min=0.05, eps_decay=0.9999, seed=0):
        self.name = name
        self.Q = np.zeros((N_STATES, N_ACTIONS))
        self.gamma, self.alpha = gamma, alpha
        self.eps, self.eps_min, self.eps_decay = eps0, eps_min, eps_decay
        self.rng = np.random.default_rng(seed)

    def act(self, s, env=None):
        if self.rng.random() < self.eps:
            return int(self.rng.integers(N_ACTIONS))
        return int(np.argmax(self.Q[s]))

    def update(self, s, a, r, s2, info=None):
        target = r + self.gamma * self.Q[s2].max()               # Eq. (4)
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])
        self.eps = max(self.eps_min, self.eps * self.eps_decay)


class SAPMAgent(QAgent):
    """
    Proposed SAPM-RL agent. Adds to Q-learning:
      1. Compliance shield: actions the policy-enforcement layer marks as
         non-compliant for the observed context are masked (never executed).
      2. Lagrangian risk control: dual variable mu raises the risk penalty
         whenever the moving average risk exceeds the target (Eq. 6).
      3. Drift detection: Page-Hinkley test on the reward stream; on alarm
         exploration and learning rate are temporarily raised (Eq. 7).
      4. Visit-count adaptive learning rate.
    """
    def __init__(self, name="SAPM-RL (proposed)", risk_target=0.20, mu_lr=0.002,
                 ph_delta=0.02, ph_lambda=25.0, **kw):
        super().__init__(name=name, **kw)
        self.N = np.zeros((N_STATES, N_ACTIONS))
        self.mu = 0.0
        self.risk_target, self.mu_lr = risk_target, mu_lr
        self.avg_risk = 0.0
        self.ph_delta, self.ph_lambda = ph_delta, ph_lambda
        self._ph_reset()
        self.drift_alarms = []
        self.t = 0

    def _ph_reset(self):
        self.ph_mean, self.ph_n, self.ph_m, self.ph_M = 0.0, 0, 0.0, 0.0

    def allowed(self, s, env):
        ctx = decode(s)
        mask = np.array([env.violation(a, ctx) == 0 for a in range(N_ACTIONS)])
        mask[0] = True
        return mask

    def act(self, s, env=None):
        mask = self.allowed(s, env) if env is not None else np.ones(N_ACTIONS, bool)
        idx = np.flatnonzero(mask)
        if self.rng.random() < self.eps:
            return int(self.rng.choice(idx))
        q = np.where(mask, self.Q[s], -np.inf)
        return int(np.argmax(q))

    def update(self, s, a, r, s2, info=None):
        self.t += 1
        risk = info["risk"] if info else 0.0
        shaped = r - self.mu * risk                                    # Eq. (6)
        self.N[s, a] += 1
        alpha = max(self.alpha, 1.0 / (1.0 + 0.2 * self.N[s, a]))
        target = shaped + self.gamma * self.Q[s2].max()
        self.Q[s, a] += alpha * (target - self.Q[s, a])
        # dual ascent on the risk constraint
        self.avg_risk = 0.99 * self.avg_risk + 0.01 * risk
        self.mu = max(0.0, self.mu + self.mu_lr * (self.avg_risk - self.risk_target))
        # Page-Hinkley drift test (detects a drop in mean reward)
        self.ph_n += 1
        self.ph_mean += (r - self.ph_mean) / self.ph_n
        self.ph_m += self.ph_mean - r - self.ph_delta
        self.ph_M = min(self.ph_M, self.ph_m) if self.ph_n > 1 else self.ph_m
        if self.ph_n > 500 and self.ph_m - self.ph_M > self.ph_lambda:
            self.drift_alarms.append(self.t)
            self.eps = max(self.eps, 0.3)
            self.N *= 0.2  # forget part of the visit history -> faster relearning
            self._ph_reset()
        self.eps = max(self.eps_min, self.eps * self.eps_decay)
