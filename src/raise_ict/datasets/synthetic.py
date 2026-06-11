"""Synthetic mini dataset used for tests and smoke runs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticDatasetSpec:
    dataset_id: str = "synthetic_raise_ict"
    n_samples: int = 240
    seed: int = 7


def load_synthetic_frame(spec: SyntheticDatasetSpec) -> pd.DataFrame:
    """Create a deterministic flow-like dataset with metadata and labels."""
    rng = np.random.default_rng(spec.seed)
    n = spec.n_samples
    flow_duration = rng.gamma(shape=2.0, scale=0.8, size=n)
    fwd_packets = rng.poisson(lam=8.0, size=n).astype(float) + 1.0
    bwd_packets = rng.poisson(lam=6.0, size=n).astype(float) + 1.0
    fwd_bytes = fwd_packets * rng.normal(loc=420.0, scale=80.0, size=n)
    bwd_bytes = bwd_packets * rng.normal(loc=360.0, scale=70.0, size=n)
    iat_mean = flow_duration / (fwd_packets + bwd_packets)
    protocol = rng.choice(["tcp", "udp"], size=n, p=[0.72, 0.28])
    service = rng.choice(["http", "dns", "ssh"], size=n, p=[0.58, 0.28, 0.14])

    logit = (
        0.010 * fwd_bytes
        - 0.007 * bwd_bytes
        + 0.95 * (service == "ssh")
        + 0.55 * (protocol == "udp")
        + rng.normal(0.0, 0.65, size=n)
    )
    threshold = np.quantile(logit, 0.68)
    label = (logit > threshold).astype(int)
    attack_type = np.where(label == 1, rng.choice(["dos", "scan"], size=n), "benign")
    groups = np.array([f"day_{idx % 4}" for idx in range(n)])

    return pd.DataFrame(
        {
            "flow_duration": flow_duration,
            "fwd_packets": fwd_packets,
            "bwd_packets": bwd_packets,
            "fwd_bytes": np.maximum(fwd_bytes, 1.0),
            "bwd_bytes": np.maximum(bwd_bytes, 1.0),
            "iat_mean": iat_mean,
            "protocol": protocol,
            "service": service,
            "timestamp": np.arange(n),
            "src_ip": [f"10.0.0.{idx % 16}" for idx in range(n)],
            "dst_ip": [f"172.16.0.{idx % 12}" for idx in range(n)],
            "attack_type": attack_type,
            "group": groups,
            "label": label,
        }
    )

