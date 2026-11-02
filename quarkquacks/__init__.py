"""QuarkQuacks: goose delivery bots for campus, assigned with QAOA and ML warm-starts."""

from .baseline import solve_classical
from .qubo import PENALTY, build_qubo, normalized_cost_vector
from .sim import Campus

__all__ = ["Campus", "build_qubo", "normalized_cost_vector", "solve_classical", "PENALTY"]
