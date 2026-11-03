"""GPU benchmark: time the same QAOA circuit on CPU vs. GPU as the problem grows.

For 2-5 geese (4, 9, 16, 25 qubits) this builds the real goose-delivery QUBO,
turns it into a p=1 QAOA circuit with fixed angles, and times one simulator run
on each device (median of several repeats, after a warm-up run).

Setup (NVIDIA GPU, Linux or WSL; native Windows and Macs are not supported):
    python -m venv gpu-env && source gpu-env/bin/activate
    pip install -r requirements.txt
    pip uninstall -y qiskit-aer && pip install qiskit-aer-gpu

Run:
    python scripts/gpu_benchmark.py                 # 2-5 geese, CPU and GPU
    python scripts/gpu_benchmark.py --geese 2 3     # quick check
    python scripts/gpu_benchmark.py --plot-only     # redraw the chart from the CSV

Outputs data/gpu_benchmark.csv and figures/gpu_benchmark.png. Without a GPU it
still runs on CPU only, so the script can be tested on any laptop.
"""

import argparse
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qiskit import transpile  # noqa: E402
from qiskit.circuit.library import QAOAAnsatz  # noqa: E402
from qiskit_aer import AerSimulator  # noqa: E402

from quarkquacks import Campus  # noqa: E402
from quarkquacks.qubo import build_qubo  # noqa: E402

CSV = "data/gpu_benchmark.csv"
FIG = "figures/gpu_benchmark.png"
# Fixed angles: the benchmark measures simulation speed, not answer quality.
# (Predicting angles at 5 geese would need the full 2^25 cost table, which is slow.)
GAMMA, BETA = 0.4, 0.6


def build_circuit(n_geese, seed):
    """Real goose-delivery QAOA circuit with n_geese^2 qubits."""
    scenario = Campus().make_scenario(n_geese, random.Random(seed))
    operator, _ = build_qubo(scenario["cost"]).to_ising()
    ansatz = QAOAAnsatz(operator, reps=1)
    circuit = ansatz.assign_parameters([BETA, GAMMA])  # Qiskit order: [beta, gamma]
    circuit.measure_all()
    return transpile(circuit, basis_gates=["h", "rx", "rz", "rzz", "cx"], optimization_level=1)


def time_run(circuit, device, shots, repeats, precision):
    sim = AerSimulator(method="statevector", device=device, precision=precision)
    sim.run(circuit, shots=shots).result()  # warm-up: loads libraries, sets up the GPU
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        sim.run(circuit, shots=shots).result()
        times.append(time.perf_counter() - t)
    return statistics.median(times)


def plot(csv_path=CSV, out=FIG):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(csv_path)
    colors = {"CPU": "#2F6FD6", "GPU": "#D9731A"}  # colorblind-safe pair
    ink, muted, grid = "#1F2A37", "#5B6775", "#E5E9EE"

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for device, sub in df.groupby("device"):
        sub = sub.sort_values("qubits")
        ax.plot(sub["qubits"], sub["seconds"], "-o", lw=2, ms=8, color=colors.get(device, ink),
                label=device)
        last = sub.iloc[-1]
        ax.annotate(f"{last['seconds']:.3g} s", (last["qubits"], last["seconds"]),
                    textcoords="offset points", xytext=(8, 0), va="center", fontsize=9, color=ink)

    if {"CPU", "GPU"} <= set(df["device"]):
        wide = df.pivot(index="qubits", columns="device", values="seconds")
        top = wide.index.max()
        speedup = wide.loc[top, "CPU"] / wide.loc[top, "GPU"]
        word = "faster" if speedup >= 1 else "slower"
        ax.set_title(f"At {top} qubits the GPU is {max(speedup, 1 / speedup):.1f}× {word} than the CPU",
                     loc="left", fontsize=11, color=ink)
    else:
        ax.set_title("QAOA simulation time (CPU only: no GPU found)", loc="left", fontsize=11, color=ink)

    geese = sorted(df["qubits"].unique())
    ax.set_xticks(geese, [f"{q}\n({int(q ** 0.5)} geese)" for q in geese])
    ax.set_yscale("log")
    ax.set_ylabel("seconds per circuit run (log scale)", color=muted, fontsize=9)
    ax.set_xlabel("qubits", color=muted, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(grid)
    ax.tick_params(colors=muted, labelsize=9)
    ax.grid(axis="y", color=grid, lw=0.8)
    ax.legend(frameon=False, fontsize=9, labelcolor=ink, loc="upper left")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"Saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geese", type=int, nargs="+", default=[2, 3, 4, 5],
                    help="goose counts to test (5 geese = 25 qubits, about 0.5 GB)")
    ap.add_argument("--shots", type=int, default=1000)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--precision", choices=["double", "single"], default="double",
                    help="single halves memory; use it if the GPU runs out of memory")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()

    if args.plot_only:
        return plot()

    available = AerSimulator().available_devices()
    devices = [d for d in ("CPU", "GPU") if d in available]
    if "GPU" not in devices:
        print("No GPU found by Qiskit Aer: running CPU only. "
              "On the GPU machine, install qiskit-aer-gpu (see the top of this file).")
    print(f"Devices: {', '.join(devices)}")

    rows = []
    for n in args.geese:
        circuit = build_circuit(n, args.seed)
        for device in devices:
            try:
                secs = time_run(circuit, device, args.shots, args.repeats, args.precision)
            except Exception as err:  # e.g. GPU out of memory
                print(f"  {n} geese on {device}: failed ({err})")
                continue
            rows.append({"geese": n, "qubits": n * n, "device": device, "seconds": secs,
                         "depth": circuit.depth(), "precision": args.precision})
            print(f"  {n} geese ({n * n} qubits) on {device}: {secs:.4f} s")

    import csv

    os.makedirs(os.path.dirname(CSV), exist_ok=True)
    with open(CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {CSV}")
    plot()


if __name__ == "__main__":
    main()
