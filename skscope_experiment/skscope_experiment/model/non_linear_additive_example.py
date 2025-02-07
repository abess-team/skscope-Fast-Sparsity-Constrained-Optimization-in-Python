import jax.numpy as jnp
import numpy as np
import cvxpy as cp
from sklearn.metrics.pairwise import rbf_kernel


def hsic(X, y, gamma_x=0.7, gamma_y=0.7):
    n, p = X.shape
    Gamma = np.eye(n) - np.ones((n, 1)) @ np.ones((1, n)) / n
    L = rbf_kernel(y.reshape(-1, 1), gamma=gamma_y)
    L_bar = Gamma @ L @ Gamma
    response = L_bar.reshape(-1)
    K_bar = np.zeros((n**2, p))
    for k in range(p):
        x = X[:, k]
        tmp = rbf_kernel(x.reshape(-1, 1), gamma=gamma_x)
        K_bar[:, k] = (Gamma @ tmp @ Gamma).reshape(-1)
    covariate = K_bar

    return covariate, response


def data_generator(n, p, sparsity_level, seed):
    """
    $y=-2\sin(2X_1)+X_2^2+X_3+\exp(-X_4)+4\sin(X_5)/(2-\sin(X_5)+\epsilon$

    Note that (X, y) is the observed samples,
    and the returned value is the processed samples.
    This is for the convenience of the experiment, 
    which is avoiding the repeated computation.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, p))
    noise = rng.normal(0, 1, n)

    true_params = np.zeros(p)  # just for indicating the support set
    true_support_set = rng.choice(np.arange(p), sparsity_level, replace=False)
    true_support_set_list = np.split(true_support_set, 5)
    for i in range(5):
        true_params[true_support_set_list[i]] = i + 1.0

    y = (
        -2 * np.sum(np.sin(2 * X[:, true_support_set_list[0]]), axis=1)
        + np.sum(np.square(X[:, true_support_set_list[1]]), axis=1)
        + np.sum(X[:, true_support_set_list[2]], axis=1)
        + np.sum(np.exp(-X[:, true_support_set_list[3]]), axis=1)
        + 4 * np.sum(np.sin(X[:, true_support_set_list[4]]) / (2 - np.sin(X[:, true_support_set_list[4]])), axis=1)
        + noise
    )

    return true_params, hsic(X, y)


def loss_jax(params, data):
    return jnp.mean(jnp.square(data[1] - data[0] @ jnp.abs(params)))


def loss_cvxpy(params, data):
    return cp.sum_squares(data[1] - data[0] @ params) / len(data[1])

def cvxpy_constraints(params):
    return [params >= 0.0]

if __name__ == "__main__":
    true_params, data = data_generator(20, 10, 8, 0)
    print(true_params)
    print(loss_jax(jnp.array(true_params), data))
    true_params_cvxpy = cp.Variable(len(true_params))
    true_params_cvxpy.value = true_params
    print(loss_cvxpy(true_params_cvxpy, data).value)
