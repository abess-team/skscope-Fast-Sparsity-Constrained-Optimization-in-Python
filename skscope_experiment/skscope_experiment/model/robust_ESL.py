import jax.numpy as jnp
import jax
import numpy as np
import cvxpy as cp


def data_generator(n, p, k, seed):
    coef = np.zeros(p)
    np.random.seed(seed)
    coef[np.random.choice(np.arange(p), k, replace=False)] = np.random.choice(
        [1, -1], k
    )
    R = np.zeros((p, p))
    for i in range(p):
        for j in range(i, p):
            R[i, j] = 0.2 ** abs(i - j)
    R = R + R.T - np.identity(p)

    x = np.random.multivariate_normal(mean=np.zeros(p), cov=R, size=(n,))
    y = np.matmul(x, coef) + np.random.randn(n)


    flip_prob = np.random.rand(len(y))
    y[flip_prob < 0.1] = -y[flip_prob < 0.1]

    return coef, (x, y)


def loss_jax(params, data):
    return jnp.sum(-jnp.exp(-jnp.square(data[1] - data[0] @ params) / 20.0))

def loss_cvxpy(params, data):
    return cp.sum(-cp.exp(-cp.square(data[1] - data[0] @ params) / 20.0))

if __name__ == "__main__":
    true_params, data = data_generator(20, 10, 5, 0)
    print(true_params)
    print(loss_jax(jnp.array(true_params), data))
    true_params_cvxpy = cp.Variable(len(true_params))
    true_params_cvxpy.value = true_params
    print(loss_cvxpy(true_params_cvxpy, data).value)

    print(jax.grad(loss_jax)(jnp.array(true_params), data))

  

