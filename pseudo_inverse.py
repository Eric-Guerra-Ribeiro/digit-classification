import numpy as np
import math

def modified_gram_schmidt(matrix, epsilon=1e-12):
    """
    """
    num_rows = np.shape(matrix)[1]
    q_factor = np.zeros(np.shape(matrix))
    r_factor = np.zeros([num_rows, num_rows])
    for j in range(num_rows):
        q_factor[:, j] = matrix[:, j]
        for i in range(j):
            r_factor[i, j] = np.dot(q_factor[:, j], q_factor[:, i])
            q_factor[:, j] = q_factor[:, j] - r_factor[i, j]*q_factor[:, i]
        r_factor[j, j] = np.linalg.norm(q_factor[:, j])
        if math.fabs(r_factor[j, j]) <= epsilon:
            print("Singular Matrix!")
        q_factor[:, j] = q_factor[:, j]/r_factor[j, j]
    return q_factor, r_factor
        

def pseudo_inverse(matrix, epsilon=1e-12):
    """
    """
    q_factor, r_factor = modified_gram_schmidt(matrix, epsilon)
    return np.linalg.inv(r_factor)@np.transpose(q_factor)
