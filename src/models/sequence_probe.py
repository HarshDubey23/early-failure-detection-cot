"""
WHAT: GRU sequence probe over the step sequence of a trajectory.

WHY:  Linear and MLP probes score each prefix in isolation. The sequence probe
      tests whether *dynamics* across steps (e.g. rising risk) carry extra
      signal beyond the current prefix state.

HOW:  Pure-numpy GRU-free fallback is NOT provided - this module needs torch.
      Without torch it raises a named error and the stage marks the GRU row
      NOT_RUN. With torch, a single-layer GRU reads per-step feature vectors
      and predicts the trajectory label from the final hidden state.
"""

from __future__ import annotations

import numpy as np


def fit_sequence_probe(sequences: list, labels: np.ndarray, seed: int = 42,
                       epochs: int = 10, hidden_size: int = 32):
    """Fit a GRU over per-step feature sequences.

    Args:
        sequences: List of (n_steps_i, n_features) arrays.
        labels: Trajectory-level failure labels.
        seed: Config seed.
        epochs: Training epochs (small; this is a probe, not a product).
        hidden_size: GRU hidden dimension.

    Returns:
        Trained torch module (with `.predict_proba_list` helper).
    """
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError(
            "The GRU sequence probe requires torch. Install requirements or "
            "mark this probe NOT_RUN."
        ) from exc

    torch.manual_seed(seed)

    class GRUProbe(nn.Module):
        def __init__(self, n_features: int, hidden: int):
            super().__init__()
            self.gru = nn.GRU(n_features, hidden, batch_first=True)
            self.head = nn.Linear(hidden, 1)

        def forward(self, padded, lengths):
            packed = nn.utils.rnn.pack_padded_sequence(
                padded, lengths, batch_first=True, enforce_sorted=False
            )
            _, hidden = self.gru(packed)
            return self.head(hidden[-1]).squeeze(-1)

    n_features = sequences[0].shape[1]
    model = GRUProbe(n_features, hidden_size)
    lengths = torch.tensor([s.shape[0] for s in sequences])
    max_len = int(lengths.max())
    padded = torch.zeros(len(sequences), max_len, n_features)
    for i, seq in enumerate(sequences):
        padded[i, : seq.shape[0]] = torch.tensor(seq, dtype=torch.float32)
    targets = torch.tensor(np.asarray(labels), dtype=torch.float32)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        optimiser.zero_grad()
        loss = loss_fn(model(padded, lengths), targets)
        loss.backward()
        optimiser.step()

    def predict_proba_list(seqs: list) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            lens = torch.tensor([s.shape[0] for s in seqs])
            m = int(lens.max())
            pad = torch.zeros(len(seqs), m, n_features)
            for i, seq in enumerate(seqs):
                pad[i, : seq.shape[0]] = torch.tensor(seq, dtype=torch.float32)
            logits = model(pad, lens)
            return torch.sigmoid(logits).numpy()

    model.predict_proba_list = predict_proba_list  # type: ignore[attr-defined]
    return model
