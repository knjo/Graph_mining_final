import numpy as np
import TDNE 

G = [[0, 0.09, 0, 0, 0, 0], [0.23, 0, 0, 0, 0, 0.62],
    [0, 0.06, 0, 0, 0, 0], [0.77, 0, 0.63, 0, 0, 0],
    [0, 0, 0, 0.65, 0, 0.38], [0, 0.85, 0.37, 0.35, 1.0, 0]]

k_step = TDNE.findK_Step(G , 3)
cp, lbd, epoch = TDNE.cp_als(k_step, R=5, max_iter=100)
Z = TDNE.tdne(cp)

print(Z)