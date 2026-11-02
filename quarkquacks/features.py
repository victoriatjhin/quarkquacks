"""Scenario features used by the warm-start model."""

import numpy as np

FEATURES = ["n", "cost_mean", "cost_std", "cost_min", "cost_max", "priority_mean", "spread"]


def scenario_features(scenario):
    """The numbers the ML model sees for one scenario, as a dict."""
    cost = np.array(scenario["cost"], dtype=float)
    priorities = [o["priority"] for o in scenario["orders"]]
    return {
        "n": len(cost),
        "cost_mean": round(float(cost.mean()), 3),
        "cost_std": round(float(cost.std()), 3),
        "cost_min": float(cost.min()),
        "cost_max": float(cost.max()),
        "priority_mean": round(float(np.mean(priorities)), 3),
        # how much better the closest goose is than the average goose, per order
        "spread": round(float((cost.mean(axis=0) - cost.min(axis=0)).mean()), 3),
    }
