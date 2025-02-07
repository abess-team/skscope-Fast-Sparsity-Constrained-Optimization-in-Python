import jax.numpy as jnp
import numpy as np
import cvxpy as cp
import _skscope_experiment


def data_generator(n, p, k, seed, rho=0.2, snr=6):
    coef = np.zeros(p)
    np.random.seed(seed)
    coef[np.random.choice(np.arange(p), k, replace=False)] = np.random.choice(
        [100, -100], k
    )
    R = np.zeros((p, p))
    for i in range(p):
        for j in range(i, p):
            R[i, j] = rho ** abs(i - j)
    R = R + R.T - np.identity(p)

    x = np.random.multivariate_normal(mean=np.zeros(p), cov=R, size=(n,))
    y = np.matmul(x, coef)
    power = np.mean(np.square(y))
    npower = power / snr
    noise = np.random.randn(len(y)) * np.sqrt(npower)
    y += noise
    y /= np.sqrt(np.mean(np.square(y)))

    return coef, (x, y)


def data_cpp_wrapper(data):
    return _skscope_experiment.RegressionData(data[0], data[1])


def loss_cpp(params, data):
    return _skscope_experiment.linear_loss(params, data)


def grad_cpp(params, data):
    return _skscope_experiment.linear_grad(params, data)


def loss_jax(params, data):
    return jnp.sum(jnp.square(data[1] - data[0] @ params))


def loss_cvxpy(params, data):
    return cp.sum_squares(data[1] - data[0] @ params)


if __name__ == "__main__":
    true_params, data = data_generator(20, 10, 5, 0)
    print(true_params)
    print(loss_jax(jnp.array(true_params), data))
    true_params_cvxpy = cp.Variable(len(true_params))
    true_params_cvxpy.value = true_params
    print(loss_cvxpy(true_params_cvxpy, data).value)
    import jax

    print(jax.grad(loss_jax)(jnp.array(true_params), data))
    print(grad_cpp(true_params, data_cpp_wrapper(data)))
