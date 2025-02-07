import jax.numpy as jnp
import numpy as np
import cvxpy as cp
from sklearn.metrics.pairwise import rbf_kernel
import _skscope_experiment


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
    $y=X_1\exp(2X_2)+X_3^2+(2X_4-1)(2X_5-1)+\epsilon$

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
        np.sum(
            X[:, true_support_set_list[0]] * np.exp(2 * X[:, true_support_set_list[1]]),
            axis=1,
        )
        + np.sum(np.square(X[:, true_support_set_list[2]]), axis=1)
        + np.sum(
            (2 * X[:, true_support_set_list[3]] - 1)
            * (2 * X[:, true_support_set_list[4]] - 1),
            axis=1,
        )
        + noise
    )

    return true_params, hsic(X, y)


def loss_jax(params, data):
    return jnp.sum(jnp.square(data[1] - data[0] @ jnp.abs(params)))


def loss_cvxpy(params, data):
    return cp.sum_squares(data[1] - data[0] @ params)


def cvxpy_constraints(params):
    return [params >= 0.0]


def data_cpp_wrapper(data):
    return _skscope_experiment.RegressionData(data[0], data[1])


def loss_cpp(params, data):
    return _skscope_experiment.positive_loss(params, data)


def grad_cpp(params, data):
    return _skscope_experiment.positive_grad(params, data)


if __name__ == "__main__":
    true_params, data = data_generator(20, 10, 5, 0)
    print(true_params)
    print(loss_jax(jnp.array(true_params), data))
    true_params_cvxpy = cp.Variable(len(true_params))
    true_params_cvxpy.value = true_params
    print(loss_cvxpy(true_params_cvxpy, data).value)
    print(loss_cpp(true_params, data_cpp_wrapper(data)))

    import jax

    print(jax.grad(loss_jax)(jnp.array(true_params), data))
    print(grad_cpp(true_params, data_cpp_wrapper(data)))
