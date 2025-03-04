from scipy.sparse import diags
import numpy as np

def JacobiSolve(M, y, x, maxIt = 10, tol = 1e-6):
    """Solve Mx = y for x
    M : a scipy.sparse.diags matrix (nx by nx)
    y : a numpy array, size nx, the RHS
    x : a numpy array, size nx, the starting point and solution
    Returns x"""
    norm = np.sqrt(np.sum(y**2))
    resid = np.sqrt(np.sum((M@x - y)**2))/norm
    
    # Split matrix M into the diagonal and off-diagonal
    D = M.diagonal()
    Moff = M.copy()
    Moff.setdiag(np.zeros(len(D)))
    for it in range(maxIt):
        if resid <= tol:
            break
        x[:] = (y - Moff@x)/D
        resid = np.sqrt(np.sum((M@x - y)**2))/norm
    else:
        print('JacobiSolve to tolerance', resid, '>', tol,
              'in', maxIt, 'iterations')
    return x
