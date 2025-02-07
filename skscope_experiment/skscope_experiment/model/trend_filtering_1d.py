import jax.numpy as jnp
import numpy as np
import cvxpy as cp
import _skscope_experiment


def data_generator(n, p, k, seed):
    # random walk with normal increment
    n = p
    np.random.seed(seed)
    noise = np.random.randn(n) / np.sqrt(n)
    true_params = np.zeros(n)
    true_params[[n // (k + 1) * i for i in range(1, k + 1)]] = 1.0
    y = np.cumsum(noise + true_params)
    return true_params, y


def loss_jax(params, data):
    return jnp.sum(jnp.square(data - jnp.cumsum(params)))


def loss_cvxpy(params, data):
    mat = np.zeros((params.size, params.size))
    mat[np.tril_indices(params.size)] = 1.0
    return cp.sum_squares(data - mat @ params) / len(data)


def data_cpp_wrapper(data):
    return _skscope_experiment.TimeSeriesData(data)


def loss_cpp(params, data):
    return _skscope_experiment.trend_filter_loss(params, data)


def grad_cpp(params, data):
    return _skscope_experiment.trend_filter_grad(params, data)


if __name__ == "__main__":
    true_params, data = data_generator(20, 20, 5, 50)
    print(true_params)
    print(loss_jax(jnp.array(true_params), data))
    true_params_cvxpy = cp.Variable(len(true_params))
    true_params_cvxpy.value = true_params
    print(loss_cvxpy(true_params_cvxpy, data).value)
    print(loss_cpp(true_params, data_cpp_wrapper(data)))
    import jax

    print(jax.grad(loss_jax)(jnp.array(true_params), data))
    print(grad_cpp(true_params, data_cpp_wrapper(data)))
