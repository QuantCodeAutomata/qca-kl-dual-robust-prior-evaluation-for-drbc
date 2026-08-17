"""
Exp 5: Closed-form benchmark for the Merton terminal-wealth neural learner.

Compares E[U(h_theta(W_T, B))] (neural) against the analytical
E[U(X_T*)] (closed-form) for four (r, b) scenarios.
"""
import os
import sys

# Ensure src/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import warnings

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from closed_form_target import (
    ALPHA,
    SIGMA,
    T,
    X_0,
    compute_closed_form_table,
    merton_terminal_wealth_samples,
)
from terminal_wealth_net import MertonBenchmarkTrainer, TerminalWealthNet

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

R_VALS = [0.05, 0.10]
B_VALS = [0.1, 0.3]
N_EXPERIMENTS = 10
N_EPOCHS = 200
N_SAMPLES = 500
GAMMA = ALPHA  # 0.5

# ---------------------------------------------------------------------------
# Step 1 – closed-form table
# ---------------------------------------------------------------------------
print("=" * 60)
print("Exp 5 – Merton Terminal-Wealth Benchmark")
print("=" * 60)

cf_table = compute_closed_form_table(R_VALS, B_VALS, SIGMA, GAMMA, X_0, T)
print("\nClosed-form expected utilities:")
print(cf_table.to_string(index=False))


# ---------------------------------------------------------------------------
# Helper: build scenario key
# ---------------------------------------------------------------------------
def _key(r, b):
    return f"r={r:.2f}_b={b:.2f}"


# ---------------------------------------------------------------------------
# Step 2 – run training experiments for each scenario
# ---------------------------------------------------------------------------
all_results = []
all_losses = {}  # scenario -> list of loss-lists

print("\nRunning training experiments …")
for r in R_VALS:
    for b in B_VALS:
        scenario = _key(r, b)
        cf_utility = float(
            cf_table.loc[(cf_table.r == r) & (cf_table.b == b), "utility_cf"].values[0]
        )
        print(f"  Scenario {scenario}  (utility_cf={cf_utility:.6f})")

        learned_utils = []
        loss_curves = []

        for exp_i in range(N_EXPERIMENTS):
            seed = 100 * int(r * 100) + int(b * 100) + exp_i
            net = TerminalWealthNet()
            trainer = MertonBenchmarkTrainer(
                net=net,
                x0=X_0,
                r=r,
                b1=b,
                b2=b,  # single scenario
                sigma=SIGMA,
                T=T,
                gamma=GAMMA,
                n_samples=N_SAMPLES,
            )
            losses, trained_net = trainer.train(n_epochs=N_EPOCHS, lr=1e-3, seed=seed)
            utility = trainer.evaluate_utility(trained_net, b=b, n_eval=2000, seed=seed)
            learned_utils.append(utility)
            loss_curves.append(losses)

        all_losses[scenario] = loss_curves
        all_results.append(
            {
                "scenario": scenario,
                "r": r,
                "b": b,
                "utility_learned_mean": float(np.mean(learned_utils)),
                "utility_learned_std": float(np.std(learned_utils)),
                "utility_cf": cf_utility,
            }
        )

results_df = pd.DataFrame(all_results)

# ---------------------------------------------------------------------------
# Step 3 – print summary table
# ---------------------------------------------------------------------------
print("\nResults summary (learned vs closed-form):")
for _, row in results_df.iterrows():
    print(
        f"  {row.scenario:20s}  "
        f"learned={row.utility_learned_mean:.6f} ± {row.utility_learned_std:.6f}  "
        f"cf={row.utility_cf:.6f}"
    )

# ---------------------------------------------------------------------------
# Step 4 – plot training loss curves (first scenario)
# ---------------------------------------------------------------------------
first_scenario = _key(R_VALS[0], B_VALS[0])
fig, ax = plt.subplots(figsize=(7, 4))
for i, losses in enumerate(all_losses[first_scenario]):
    ax.semilogy(losses, alpha=0.5, linewidth=0.8, label=f"run {i+1}" if i < 3 else None)
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE loss (log scale)")
ax.set_title(f"Training loss – scenario {first_scenario}")
ax.legend(fontsize=7)
plt.tight_layout()
loss_path = os.path.join(RESULTS_DIR, "exp5_training_loss.png")
plt.savefig(loss_path, dpi=120)
plt.close(fig)
print(f"\nSaved training loss plot → {loss_path}")

# ---------------------------------------------------------------------------
# Step 5 – scatter: learned vs true terminal wealth (first scenario)
# ---------------------------------------------------------------------------
r_s, b_s = R_VALS[0], B_VALS[0]
rng = np.random.default_rng(0)
W_T_eval = rng.normal(0.0, np.sqrt(T), size=500)
X_T_star = merton_terminal_wealth_samples(b_s, r_s, SIGMA, GAMMA, X_0, T, W_T_eval)

# Re-train one net for scatter
net_s = TerminalWealthNet()
trainer_s = MertonBenchmarkTrainer(
    net=net_s, x0=X_0, r=r_s, b1=b_s, b2=b_s, sigma=SIGMA, T=T, gamma=GAMMA, n_samples=N_SAMPLES
)
_, net_s = trainer_s.train(n_epochs=N_EPOCHS, lr=1e-3, seed=0)

B_eval = np.full(len(W_T_eval), b_s)
inputs = torch.tensor(np.stack([W_T_eval, B_eval], axis=1), dtype=torch.float32)
net_s.eval()
with torch.no_grad():
    h_learned = net_s(inputs).squeeze().numpy()

fig2, ax2 = plt.subplots(figsize=(5, 5))
ax2.scatter(X_T_star, h_learned, s=10, alpha=0.4, label="samples")
lo = min(X_T_star.min(), h_learned.min())
hi = max(X_T_star.max(), h_learned.max())
ax2.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="identity")
ax2.set_xlabel("True $X_T^*$ (closed-form)")
ax2.set_ylabel("Learned $h_\\theta(W_T, B)$")
ax2.set_title(f"Learned vs True – scenario {_key(r_s, b_s)}")
ax2.legend(fontsize=8)
plt.tight_layout()
scatter_path = os.path.join(RESULTS_DIR, "exp5_wealth_scatter.png")
plt.savefig(scatter_path, dpi=120)
plt.close(fig2)
print(f"Saved wealth scatter plot → {scatter_path}")

# ---------------------------------------------------------------------------
# Step 6 – save CSV
# ---------------------------------------------------------------------------
csv_path = os.path.join(RESULTS_DIR, "exp5_results.csv")
results_df.to_csv(csv_path, index=False)
print(f"Saved results CSV → {csv_path}")
print("\nExp 5 complete.\n")
