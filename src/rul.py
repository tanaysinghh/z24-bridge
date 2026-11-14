import numpy as np

from src.taxonomy import DSI, TIERS, tier_for

DEFAULTS = {
    "n_samples": 4000,
    "dsi_noise_sd": 0.04,
    "design_life_mean": 100.0,
    "design_life_sd": 10.0,
    "design_life_min": 60.0,
    "beta_low": 1.5,
    "beta_high": 3.0,
    "interval": 0.90,
}


def expected_dsi(probs):
    return np.asarray(probs, dtype=float) @ np.asarray(DSI)


def sample_dsi(probs, rng, n, noise_sd):
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    classes = rng.choice(len(probs), size=n, p=probs)
    return np.clip(np.asarray(DSI)[classes] + rng.normal(0.0, noise_sd, n), 0.0, 1.0)


def sample_rul(dsi, rng, params):
    n = len(dsi)
    life = np.maximum(rng.normal(params["design_life_mean"], params["design_life_sd"], n), params["design_life_min"])
    beta = rng.uniform(params["beta_low"], params["beta_high"], n)
    return life * (1.0 - dsi ** (1.0 / beta))


def summarize(x, interval):
    lo, hi = (1 - interval) / 2, 1 - (1 - interval) / 2
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "lower": float(np.quantile(x, lo)),
        "upper": float(np.quantile(x, hi)),
    }


def estimate(probs, seed=0, **overrides):
    params = {**DEFAULTS, **overrides}
    rng = np.random.default_rng(seed)
    dsi = sample_dsi(probs, rng, params["n_samples"], params["dsi_noise_sd"])
    rul = sample_rul(dsi, rng, params)
    point = float(expected_dsi(probs))
    tier = tier_for(point)
    tier_counts = np.bincount([tier_for(d) for d in dsi], minlength=len(TIERS)) / len(dsi)
    return {
        "dsi": {"expected": point, **summarize(dsi, params["interval"])},
        "rul_years": summarize(rul, params["interval"]),
        "interval": params["interval"],
        "tier": {"index": tier, **TIERS[tier]},
        "tier_probabilities": {TIERS[k]["name"]: float(v) for k, v in enumerate(tier_counts)},
        "histogram": histogram(rul),
        "params": params,
    }


def histogram(rul, bins=24):
    hi = max(float(np.quantile(rul, 0.995)), 1.0)
    counts, edges = np.histogram(rul, bins=bins, range=(0.0, hi))
    return {"edges": [float(e) for e in edges], "counts": [int(c) for c in counts]}


def deterministic_curve(params=None, points=21):
    params = {**DEFAULTS, **(params or {})}
    beta = (params["beta_low"] + params["beta_high"]) / 2
    d = np.linspace(0, 1, points)
    return {"dsi": d.tolist(), "rul_years": (params["design_life_mean"] * (1 - d ** (1 / beta))).tolist(), "beta": beta}
