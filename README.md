# QuarkQuacks 🪿

Robot geese that deliver food and drinks across campus, routed by a quantum algorithm.

## What it does

QuarkQuacks decides which robot goose delivers which order from the food court to study rooms. It solves the problem with QAOA, a quantum optimization algorithm, and uses machine learning to help QAOA find the answer faster.

## How it works

1. Generate geese and orders on a campus map.
2. Turn "which goose gets which order" into a QUBO, a problem format quantum computers can solve.
3. Solve it with QAOA in Qiskit, using ML to predict good starting settings.
4. Compare the result with a classical solver.

## Tech stack

Python, Qiskit, Qiskit Optimization, Qiskit Aer, scikit-learn, NumPy, SciPy, Matplotlib

## Project structure

| Folder | Contents |
| --- | --- |
| `quarkquacks/` | Campus map, QUBO builder, classical solver |
| `quantum/` | QAOA solver |
| `ml/` | ML model |
| `demo/` | Goose animation |
| `data/`, `figures/` | Data, results, charts |
| `tests/` | Tests |
