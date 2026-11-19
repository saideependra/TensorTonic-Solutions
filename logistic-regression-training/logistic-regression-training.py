import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Write code here
    X = np.asarray(X,dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    N,D = X.shape

    w = np.zeros(D, dtype=float)
    b = 0.0

    for _ in range(steps):
        z = X @ w + b
        p = _sigmoid(z)

        err = p - y
        grad_w = X.T @ err / N
        grad_b = err.sum() / N

        w -= lr * grad_w
        b -= lr * grad_b

    return (w, float(b))