"""1D convolutional network over raw sensor windows (optional, needs PyTorch).

The boosted model sees hand-built summaries of each engine's history; this
network reads the last ``window`` cycles of normalised sensors directly. Their
errors are only partly correlated, so ``RULModel`` can blend the two, with the
weight chosen on out-of-fold predictions. Install with ``pip install turbosensor[deep]``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def _require_torch():
    try:
        import torch
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError("the CNN member needs PyTorch: pip install 'turbosensor[deep]'") from e
    return torch


def sequence_inputs(Z: pd.DataFrame, sensors: Sequence[str], baseline_cycles: int = 15) -> pd.DataFrame:
    """Per-cycle network inputs: normalised sensors, drift from the engine's
    own early-life baseline (causal, frozen after ``baseline_cycles``), and age."""
    Z = Z.sort_values(["unit", "cycle"])
    S = Z[list(sensors)]
    position = Z.groupby("unit").cumcount().to_numpy()
    base = S.groupby(Z["unit"], sort=False).expanding().mean().reset_index(level=0, drop=True).loc[Z.index]
    base[position >= baseline_cycles] = np.nan
    base = base.groupby(Z["unit"]).ffill()
    out = pd.concat([Z[["unit"]], S, (S - base).add_suffix("_drift")], axis=1)
    out["age"] = Z["cycle"] / 100.0
    return out


def windows(inputs: pd.DataFrame, length: int) -> tuple[np.ndarray, pd.Index]:
    """One window ending at every row, left-padded with the engine's first cycle."""
    cols = [c for c in inputs.columns if c != "unit"]
    blocks = []
    for _, g in inputs.groupby("unit", sort=False):
        a = g[cols].to_numpy(np.float32)
        a = np.vstack([np.repeat(a[:1], length - 1, axis=0), a])
        idx = np.arange(len(g))[:, None] + np.arange(length)[None, :]
        blocks.append(a[idx])
    return np.concatenate(blocks), inputs.index


class CNNRegressor:
    """Small temporal CNN: three conv layers, mean- and last-step pooling, MLP head."""

    def __init__(
        self,
        window: int = 60,
        epochs: int = 40,
        batch_size: int = 256,
        lr: float = 3e-3,
        target_scale: float = 125.0,
        seed: int = 0,
        threads: int | None = None,
    ):
        self.window = window
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.target_scale = target_scale
        self.seed = seed
        self.threads = threads

    def _net(self, channels: int):
        torch = _require_torch()
        nn = torch.nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv1d(channels, 32, 5, padding=2),
                    nn.ReLU(),
                    nn.Conv1d(32, 64, 5, padding=2),
                    nn.ReLU(),
                    nn.Conv1d(64, 64, 3, padding=1),
                    nn.ReLU(),
                )
                self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

            def forward(self, x):  # x: batch, time, channels
                h = self.conv(x.transpose(1, 2))
                return self.head(torch.cat([h.mean(-1), h[..., -1]], 1)).squeeze(-1)

        return Net()

    def fit(self, X: np.ndarray, y: np.ndarray) -> CNNRegressor:
        torch = _require_torch()
        if self.threads:
            torch.set_num_threads(self.threads)
        torch.manual_seed(self.seed)
        net = self._net(X.shape[2])
        steps = self.epochs * int(np.ceil(len(X) / self.batch_size))
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=steps)
        Xt = torch.from_numpy(X)
        yt = torch.from_numpy((np.asarray(y) / self.target_scale).astype(np.float32))
        gen = torch.Generator().manual_seed(self.seed)
        for _ in range(self.epochs):
            net.train()
            perm = torch.randperm(len(Xt), generator=gen)
            for i in range(0, len(Xt), self.batch_size):
                b = perm[i : i + self.batch_size]
                loss = ((net(Xt[b]) - yt[b]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
        net.eval()
        # Keep weights only, so the fitted model pickles without live torch state.
        self.channels_ = X.shape[2]
        self.state_ = {k: v.detach().clone() for k, v in net.state_dict().items()}
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        torch = _require_torch()
        net = self._net(self.channels_)
        net.load_state_dict(self.state_)
        net.eval()
        with torch.no_grad():
            out = [net(torch.from_numpy(X[i : i + 4096])) for i in range(0, len(X), 4096)]
        return self.target_scale * torch.cat(out).numpy()
