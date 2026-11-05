import random
from ml.model import predict_angles
from quarkquacks import Campus
from quantum.solver import solve_qaoa
from quarkquacks import solve_classical

scn = Campus().make_scenario(3, random.Random(0))
angles = predict_angles(scn)
print(angles)                                   # expect two numbers: (gamma, beta)
print(solve_qaoa(scn["cost"], warm_start=angles))
print("Classical:", solve_classical(scn["cost"]))

import numpy as np
print(np.array(scn["cost"]))