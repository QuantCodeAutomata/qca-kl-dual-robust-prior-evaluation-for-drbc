"""
Neural learner for Merton terminal-wealth function.
Based on Theorem 4.1 of the DRBC paper.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from closed_form_target import merton_expected_utility, merton_terminal_wealth_samples


class TerminalWealthNet(nn.Module):
    """
    Feed-forward network h_theta(W_T, B) -> scalar terminal wealth.

    Architecture:
        Input (2) -> Linear(128) -> LeakyReLU -> Linear(256) -> LeakyReLU
                  -> Linear(256) -> LeakyReLU -> Linear(1)
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_sizes: list = None,
        output_dim: int = 1,
        leaky_slope: float = 0.01,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [128, 256, 256]
        layers = []
        in_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.LeakyReLU(leaky_slope))
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, 2) tensor  –  columns are [W_T, B]

        Returns
        -------
        (batch, 1) tensor  –  h_theta(W_T, B)
        """
        return self.net(x)


class MertonBenchmarkTrainer:
    """Train a TerminalWealthNet to match the closed-form Merton solution."""

    def __init__(
        self,
        net: TerminalWealthNet,
        x0: float = 1.0,
        r: float = 0.05,
        b1: float = 0.1,
        b2: float = 0.3,
        sigma: float = 0.4,
        T: float = 1.0,
        gamma: float = 0.5,
        n_samples: int = 500,
        device: str = "cpu",
    ):
        self.net = net
        self.x0 = x0
        self.r = r
        self.b1 = b1
        self.b2 = b2
        self.sigma = sigma
        self.T = T
        self.gamma = gamma
        self.n_samples = n_samples
        self.device = device

    # ------------------------------------------------------------------
    # Data generation
    # ------------------------------------------------------------------

    def generate_samples(self, n: int, seed=None):
        """
        Sample W_T ~ N(0, T) with B fixed at b1 (Q^{b1} benchmark measure).

        Returns
        -------
        (W_T, B) : pair of (n,) float32 tensors on self.device
        """
        rng = np.random.default_rng(seed)
        W_T_np = rng.normal(0.0, np.sqrt(self.T), size=n)
        B_np = np.full(n, self.b1)
        W_T = torch.tensor(W_T_np, dtype=torch.float32, device=self.device)
        B = torch.tensor(B_np, dtype=torch.float32, device=self.device)
        return W_T, B

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        h_vals: torch.Tensor,
        W_T: torch.Tensor,
        B: torch.Tensor,
        measure: str = "b1",
    ) -> torch.Tensor:
        """
        MSE between network output h_theta and closed-form X_T*.

        Parameters
        ----------
        h_vals : (batch, 1) or (batch,) tensor – network predictions
        W_T, B : (batch,) tensors  (B currently fixed, kept for API generality)
        measure : 'b1' or 'b2' – which drift value to use for the target
        """
        W_T_np = W_T.detach().cpu().numpy()
        b = self.b1 if measure == "b1" else self.b2
        X_T_star = merton_terminal_wealth_samples(
            b=b,
            r=self.r,
            sigma=self.sigma,
            gamma=self.gamma,
            x0=self.x0,
            T=self.T,
            W_T_samples=W_T_np,
        )
        target = torch.tensor(X_T_star, dtype=torch.float32, device=self.device)
        return nn.functional.mse_loss(h_vals.squeeze(), target)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, n_epochs: int = 200, lr: float = 1e-3, seed: int = 42):
        """
        Train with Adam and a 3-phase LR schedule:
          - epochs [0, 10)     : linear warmup to lr
          - epochs [10, 150)   : stable at lr
          - epochs [150, end)  : linear decay towards 3e-4

        Returns
        -------
        (losses, net) : list of per-epoch loss floats and the trained network
        """
        net = self.net.to(self.device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        warmup_end = 10
        stable_end = warmup_end + 140
        final_lr_ratio = 3e-4 / lr

        def lr_lambda(epoch):
            if epoch < warmup_end:
                return (epoch + 1) / warmup_end
            if epoch < stable_end:
                return 1.0
            decay_steps = max(n_epochs - stable_end, 1)
            step = epoch - stable_end
            return max(
                final_lr_ratio,
                1.0 - step / decay_steps * (1.0 - final_lr_ratio),
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        W_T, B = self.generate_samples(self.n_samples, seed=seed)
        inputs = torch.stack([W_T, B], dim=1)  # (n_samples, 2)

        losses = []
        net.train()
        for _ in range(n_epochs):
            optimizer.zero_grad()
            h_vals = net(inputs)
            loss = self.compute_loss(h_vals, W_T, B)
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())

        return losses, net

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_utility(
        self,
        net: TerminalWealthNet,
        b: float,
        n_eval: int = 2000,
        seed: int = 0,
    ) -> float:
        """
        Estimate E[U(h_theta(W_T, B))] = E[h_theta^gamma / gamma] under Q^b.

        Parameters
        ----------
        net   : trained network
        b     : drift value defining the evaluation measure Q^b
        n_eval: number of Monte-Carlo samples
        seed  : RNG seed

        Returns
        -------
        float – estimated expected utility
        """
        rng = np.random.default_rng(seed)
        W_T_np = rng.normal(0.0, np.sqrt(self.T), size=n_eval)
        B_np = np.full(n_eval, b)
        inputs = torch.tensor(
            np.stack([W_T_np, B_np], axis=1), dtype=torch.float32, device=self.device
        )

        net.eval()
        with torch.no_grad():
            h_vals = net(inputs).squeeze().cpu().numpy()

        h_vals = np.maximum(h_vals, 1e-8)  # guard against non-positive outputs
        return float(np.mean(h_vals**self.gamma / self.gamma))

    # ------------------------------------------------------------------
    # Benchmark runner
    # ------------------------------------------------------------------

    def run_benchmark(self, n_experiments: int = 10, seed: int = 0) -> pd.DataFrame:
        """
        Run n_experiments independent training runs for both b1 and b2.

        Returns DataFrame with columns:
            b, utility_cf, utility_learned_mean, utility_learned_std
        """
        rows = []
        for b_val in (self.b1, self.b2):
            cf_utility = merton_expected_utility(
                b=b_val,
                r=self.r,
                sigma=self.sigma,
                gamma=self.gamma,
                x0=self.x0,
                T=self.T,
            )
            learned_utils = []
            for exp_i in range(n_experiments):
                fresh_net = TerminalWealthNet()
                trainer = MertonBenchmarkTrainer(
                    net=fresh_net,
                    x0=self.x0,
                    r=self.r,
                    b1=b_val,
                    b2=self.b2,
                    sigma=self.sigma,
                    T=self.T,
                    gamma=self.gamma,
                    n_samples=self.n_samples,
                    device=self.device,
                )
                _, trained_net = trainer.train(n_epochs=200, lr=1e-3, seed=seed + exp_i)
                learned_utils.append(self.evaluate_utility(trained_net, b=b_val))
            rows.append(
                {
                    "b": b_val,
                    "utility_cf": cf_utility,
                    "utility_learned_mean": float(np.mean(learned_utils)),
                    "utility_learned_std": float(np.std(learned_utils)),
                }
            )
        return pd.DataFrame(rows)
