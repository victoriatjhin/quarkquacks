# QuarkQuacks 🪿

**Quantum-assigned goose delivery bots for campus.**

A fleet of robot geese, Waterloo's most famous residents, delivers coffee, pizza and bubble tea from the food court to study rooms. QuarkQuacks decides which goose takes which order by turning the problem into a QUBO and solving it with QAOA in Qiskit. A machine-learning model predicts good QAOA starting angles, so the quantum solver reaches a good plan in fewer steps. Every result is checked against an exact classical solver.

Built for **Qiskit Fall Fest 2026** at the University of Waterloo (November 1–6, 2026).

> **Hackathon prompt:** create a tool or abstraction using Qiskit that bridges a problem in a non-quantum field to a quantum algorithm or application.

## The problem

Orders pile up at the food court while geese are scattered around campus. Every order needs exactly one goose, and every goose takes one order. Hot coffee should arrive before bubble tea, and nobody wants a goose waddling across campus when another is closer.

Choosing the best goose-to-order pairing is an **assignment problem**. It maps naturally to a QUBO (quadratic unconstrained binary optimization), the form that quantum optimization algorithms like QAOA solve.

## How it works

```
Campus map + orders  →  travel costs  →  QUBO  →  QAOA (Qiskit)  →  goose assignments
                                                     ↑
                                    ML-predicted starting angles
```

1. **Campus simulator:** a grid map with food counters (coffee, pizza, bubble tea), hallways and study rooms. Geese and orders are placed at random, and hot items get higher priority.
2. **Travel costs:** shortest-path walking distance (goose → counter → room), weighted by the order's priority.
3. **Bridge to quantum:** the assignment problem becomes a QUBO with `qiskit-optimization`.
4. **Quantum solve:** QAOA on the Qiskit Aer simulator.
5. **ML warm-start:** a model trained on past scenarios predicts good QAOA angles, so the optimizer starts close to a good answer instead of from random guesses.
6. **Classical baseline:** an exact solver (`scipy.optimize.linear_sum_assignment`) gives the true best answer for comparison.

## Why ML?

QAOA has settings (angles) that normally need many rounds of trial and error to tune for each new problem. Our model learns from hundreds of simulated scenarios and predicts good angles instantly. We measure the benefit as fewer optimizer steps and a higher chance of finding the best plan.

## Results

*To be updated by Friday.* Early prototype runs suggest ML warm-starting cuts QAOA optimizer steps substantially for 3 geese (9 qubits), with the same solution quality.

We don't claim QAOA beats the classical solver: at this size the classical solver is exact and instant. QuarkQuacks shows a clean pipeline from a real-world problem to a quantum algorithm, and how ML can make that algorithm cheaper to run.

## Planned repository layout

| Path | Contents |
| --- | --- |
| `quarkquacks/` | Shared code: campus simulator, QUBO builder, classical baseline |
| `quantum/` | Qiskit QAOA solver |
| `ml/` | Warm-start model training and evaluation |
| `scripts/` | Data generation |
| `demo/` | Campus map plots and the animated goose demo |
| `data/` | Generated scenarios and results |
| `tests/` | Unit tests |

## Getting started

*Coming soon.* Requirements: Python 3.10+, `qiskit`, `qiskit-optimization`, `qiskit-algorithms`, `qiskit-aer`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`.

## Team

| Name | Role |
| --- | --- |
| *Name* | Campus simulator and goose demo |
| *Name* | Quantum solver (QAOA in Qiskit) |
| *Name* | ML warm-start, benchmarks and presentation |

## Roadmap

- [ ] Campus simulator and order generator
- [ ] Assignment → QUBO bridge
- [ ] QAOA solver on the Aer simulator
- [ ] Classical baseline
- [ ] ML warm-start model and benchmarks
- [ ] Animated goose delivery demo
- [ ] Stretch: run the smallest case on real IBM quantum hardware
- [ ] Stretch: GPU vs. CPU simulation benchmark
