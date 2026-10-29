# SAPM-RL: Self-Adaptive Privacy Management using Reinforcement Learning

Source code for the case study "Self-Adaptive Privacy Management Using Reinforcement Learning"
(Team 26FSDPYCPMID1032, Nandini Rana, 23BCE0049).

## Files

| File | Purpose |
|---|---|
| `sapm_env.py` | Privacy environment: WDBC query pool, context, actions, risk (Eq. 8), utility (Eq. 11), compliance rules, budget accountant, threat dynamics, concept drift |
| `sapm_agents.py` | Static, rule-based, contextual-bandit, Q-learning and the proposed SAPM-RL agent (shield, Lagrangian risk control, Page-Hinkley drift detection) |
| `run_experiments.py` | Main experiment (6 methods x 5 seeds x 60,000 requests), privacy-utility sweep, robustness test, explainability analysis |
| `figs_concept.py` | Architecture and workflow figures (600 DPI) |
| `figs_results.py` | Result figures (600 DPI) |

## Requirements

Python 3.10+ with `numpy`, `pandas`, `scikit-learn`, `matplotlib`.

```
pip install numpy pandas scikit-learn matplotlib
```

## Reproduce

```
python run_experiments.py   # about 8 minutes on 2 CPU cores; writes ../results and ../data
python figs_concept.py      # writes ../figs
python figs_results.py      # writes ../figs
```

## Data

The Wisconsin Diagnostic Breast Cancer dataset (569 records, 30 attributes) is loaded from scikit-learn
(`sklearn.datasets.load_breast_cancer`); original source: UCI Machine Learning Repository,
https://doi.org/10.24432/C5DW2B. `run_experiments.py` also exports it to `../data/wdbc_health_records.csv`
together with the generated query pool (`../data/query_pool.csv`).

## Outputs (`../results`)

- `summary_all_seeds.csv`, `summary_mean_std.csv` - main results (Table 11)
- `curves.npz` - rolling reward, violation and agreement curves (Figure 24)
- `tradeoff.csv` - privacy-utility sweep (Figure 25)
- `robustness.csv` - observation-noise test (Table A1)
- `fairness_groups.csv` - utility and denial by group (Table A2)
- `explainability.json`, `sapm_Q.npy`, `sapm_meta.json` - learned policy and explanation statistics (Figure 18)
- `actions_*.npy` - per-request actions for seed 0 (Figure 16)
