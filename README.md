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

## How to run

```bash
git clone https://github.com/victoriatjhin/quarkquacks.git
cd quarkquacks
pip install -r requirements.txt
```

Then:

```bash
python run_pipeline.py --geese 3       # full pipeline on one scenario, saves a plan plot
python scripts/generate_data.py        # regenerate training data (about 30 s)
python -m ml.model                     # train the warm-start model
python -m ml.warmstart                 # compare random vs. warm-start QAOA
python scripts/scaling.py              # scaling study at 2-4 geese (about 1.5 min)
python scripts/plot_scaling.py         # draw figures/scaling.png
pytest                                 # run tests
```

## Demo

Goose delivery animation and results charts coming soon.

## Project structure

| Folder | Contents |
| --- | --- |
| `quarkquacks/` | Campus map, QUBO builder, classical solver |
| `quantum/` | QAOA solver |
| `ml/` | ML model |
| `demo/` | Goose animation |
| `data/`, `figures/` | Data, results, charts |
| `tests/` | Tests |
