"""Train the warm-start model and predict QAOA angles for new scenarios.

    python -m ml.model          # train both models, keep the better one, save it

Other code only needs:

    from ml.model import predict_angles
    gamma, beta = predict_angles(scenario)
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

from quarkquacks.features import FEATURES, scenario_features
from quarkquacks.qaoa_sim import energy
from quarkquacks.qubo import normalized_cost_vector

TARGETS = ["best_gamma", "best_beta"]
MODEL_PATH = os.path.join(os.path.dirname(__file__), "warmstart_model.joblib")
DATA_DIR = "data"

CANDIDATES = {
    "k-nearest neighbours": lambda: KNeighborsRegressor(n_neighbors=5),
    "random forest": lambda: RandomForestRegressor(n_estimators=200, random_state=0),
}


def load_split(seed=0):
    """80/20 split, stratified by size, with a fixed seed so results are repeatable."""
    df = pd.read_csv(os.path.join(DATA_DIR, "dataset.csv"))
    return train_test_split(df, test_size=0.2, random_state=seed, stratify=df["n"])


def mean_start_energy(model, test, scenarios):
    """Average QAOA energy at the predicted angles (lower = better starting point)."""
    predicted = model.predict(test[FEATURES])
    energies = []
    for (_, row), (g, b) in zip(test.iterrows(), predicted):
        C = normalized_cost_vector(scenarios[row["id"]]["cost"])
        energies.append(energy(C, g, b))
    return float(np.mean(energies))


def train_and_select(seed=0, verbose=True):
    """Train each candidate on the training split and keep the one with the lowest test energy."""
    train, test = load_split(seed)
    with open(os.path.join(DATA_DIR, "scenarios.json")) as f:
        scenarios = {s["id"]: s for s in json.load(f)}

    results = {}
    for name, make in CANDIDATES.items():
        model = make().fit(train[FEATURES], train[TARGETS])
        angle_error = np.abs(model.predict(test[FEATURES]) - test[TARGETS].to_numpy()).mean()
        results[name] = (model, mean_start_energy(model, test, scenarios), angle_error)
        if verbose:
            print(f"{name:22s} mean start energy {results[name][1]:.4f} | "
                  f"mean angle error {angle_error:.4f}")

    best_name = min(results, key=lambda k: results[k][1])
    joblib.dump({"name": best_name, "model": results[best_name][0], "features": FEATURES}, MODEL_PATH)
    if verbose:
        print(f"Kept: {best_name} (saved to ml/warmstart_model.joblib)")
    return best_name, results


_cached = None


def predict_angles(scenario):
    """Predicted (gamma, beta) for p=1 QAOA, in the normalised-cost convention of qaoa_sim."""
    global _cached
    if _cached is None:
        _cached = joblib.load(MODEL_PATH)
    x = pd.DataFrame([scenario_features(scenario)])[_cached["features"]]
    gamma, beta = _cached["model"].predict(x)[0]
    return float(gamma), float(beta)


if __name__ == "__main__":
    train_and_select()
