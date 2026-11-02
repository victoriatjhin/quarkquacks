# QuarkQuacks 🪿

**Quantum-assigned goose delivery bots for campus.**

Robot geese deliver coffee, pizza and bubble tea from the food court to study rooms. QuarkQuacks assigns geese to orders by turning the problem into a QUBO and solving it with QAOA in Qiskit. An ML model predicts good QAOA starting angles, so the solver reaches a good plan in fewer steps.

## How it works

```
Campus map + orders → travel costs → QUBO → QAOA (Qiskit) → assignments
                                              ↑
                                  ML-predicted starting angles
```

1. **Simulate:** random geese and orders on a campus grid; hot items get higher priority.
2. **Bridge:** convert the goose-to-order assignment into a QUBO with `qiskit-optimization`.
3. **Solve:** run QAOA on the Qiskit Aer simulator, warm-started by the ML model.
4. **Compare:** check against an exact classical solver.

## Results

*Coming Friday.* We measure how much ML warm-starting reduces QAOA optimizer steps. We don't claim quantum beats classical; at this size the classical solver is exact and instant.

## Structure

| Folder | Contents |
| --- | --- |
| `quarkquacks/` | Simulator, QUBO builder, classical baseline |
| `quantum/` | QAOA solver |
| `ml/` | Warm-start model |
| `demo/` | Goose animation |
| `data/`, `figures/` | Scenarios, results, charts |
| `tests/` | Unit tests |
