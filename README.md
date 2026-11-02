# QuarkQuacks 🪿

Robot geese deliver food and drinks from the food court to study rooms. QuarkQuacks uses a quantum algorithm (QAOA) to decide which goose delivers which order, and machine learning to help it find the answer faster.

## How it works

1. Generate geese and orders on a campus map.
2. Turn "which goose gets which order" into a QUBO, a problem format quantum computers can solve.
3. Solve it with QAOA in Qiskit, using ML to predict good starting settings.
4. Compare the result with a classical solver.

## Structure

| Folder | Contents |
| --- | --- |
| `quarkquacks/` | Campus map, QUBO builder, classical solver |
| `quantum/` | QAOA solver |
| `ml/` | ML model |
| `demo/` | Goose animation |
| `data/`, `figures/` | Data, results, charts |
| `tests/` | Tests |
