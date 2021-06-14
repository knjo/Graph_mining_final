from typing import List
import numpy as np
import tensorly as tl
from tensorly.decomposition._cp import initialize_cp
from tensorly.tenalg import khatri_rao
from tqdm import tqdm
 
# Function to multiply two matrices A and B
def multiply(A: List[List[float]], B: List[List[float]],
             N: int) -> List[List[float]]:
    C = [[0 for _ in range(N)] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i][j] += A[i][k] * B[k][j]
    return C
 
# Function to calculate the power of a matrix
def matrix_power(M: List[List[float]], p: int, n: int) -> List[List[float]]:
    A = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        A[i][i] = 1
    while (p):
        if (p % 2):
            A = multiply(A, M, n)
        M = multiply(M, M, n)
        p //= 2
    return A
 
# Function to calculate the probability of
# reaching F at time T after starting from S
def findProbability(M: List[List[float]], N: int, F: int, S: int,
                    T: int) -> float:
 
    # Storing M^T in MT
    MT = matrix_power(M, T, N)
 
    # Returning the answer
    return MT[F - 1][S - 1]

def findK_Step( adj : List[List[float]] , K_step : int):
    
    N = len(adj)
    k_step = np.zeros((K_step,N,N))
    for i in range (K_step):
        T = i
        for j in range(N):
            S = j
            for k in range(N):
                F = k
                k_step[i][j][k] = findProbability(adj, N, F, S, T)
    return k_step
 

def cp_als(tensor: np.ndarray, R=3, max_iter=100):
    N = tl.ndim(tensor)
    print(N)
    # Step 1
    lbd, A = initialize_cp(tensor, R, init='svd', svd='numpy_svd',
                           random_state=0,
                           normalize_factors=True)
    # A = []
    # for n in range(N):
    #     np.random.seed(N)
    #     A.append(tl.tensor(np.random.random((tensor.shape[n], rank))))
    # lbd = tl.ones(rank)

    for epoch in tqdm(range(max_iter)):
        for n in range(N):
            V = np.ones((R, R))
            for i in range(N):
                if i != n:
                    V = np.matmul(A[i].T, A[i]) * V

            T = khatri_rao(A, skip_matrix=n)
            A[n] = np.matmul(np.matmul(tl.unfold(tensor, mode=n), T), np.linalg.pinv(V))

            for r in range(R):
                lbd[r] = tl.norm(A[n][:, r])
            A[n] = A[n] / tl.reshape(lbd, (1, -1))
            
    return A, lbd, max_iter

def tdne (cp):
    C = cp[0]
    B = cp[1]
    A = cp[2]
    ST = A @ C.T
    TT = B @ C.T
    Z = np.concatenate([ST , TT])
    return Z