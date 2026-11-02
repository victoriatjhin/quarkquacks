import random

import numpy as np
from qiskit.circuit.library import QAOAAnsatz
from qiskit.primitives import StatevectorEstimator

from ml.model import predict_angles
from quantum.solver import to_qiskit_angles
from quarkquacks import Campus
from quarkquacks.qaoa_sim import energy
from quarkquacks.qubo import build_qubo, qubo_cost_vector


def test_predicted_angles_transfer_to_qiskit():
    """Model angles must give the same QAOA energy in Qiskit as in the numpy simulator."""
    scn = Campus().make_scenario(2, random.Random(3))
    qubo = build_qubo(scn["cost"])
    C = qubo_cost_vector(qubo)
    scale = np.abs(C).max()
    op, offset = qubo.to_ising()

    gamma, beta = predict_angles(scn)
    ansatz = QAOAAnsatz(op, reps=1)                  # parameters ordered [beta, gamma]
    bound = ansatz.assign_parameters(to_qiskit_angles(qubo, gamma, beta))
    e_qiskit = float(StatevectorEstimator().run([(bound, op)]).result()[0].data.evs)

    assert np.isclose((e_qiskit + offset) / scale, energy(C / scale, gamma, beta))
