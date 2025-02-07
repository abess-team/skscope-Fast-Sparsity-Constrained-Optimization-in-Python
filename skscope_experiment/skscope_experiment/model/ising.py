import numpy as np
import jax.numpy as jnp
import cvxpy as cp
import _skscope_experiment


class IsingData:
    def __init__(self, data):
        self.n = data.shape[0]
        self.p = data.shape[1] - 1
        self.table = data[:, 1:]
        self.freq = data[:, 0]

        self.index_translator = np.zeros(shape=(self.p, self.p), dtype=np.int32)
        idx = 0
        for i in range(self.p):
            for j in range(i + 1, self.p):
                self.index_translator[i, j] = idx
                self.index_translator[j, i] = idx
                idx += 1


def data_generator(n, p, k, seed, strength=0.5):
    np.random.seed(seed)
    P = int((1 + np.sqrt(1 + 8 * p)) / 2)
    if P * (P - 1) / 2 != p:
        raise ValueError("p must be a triangular number")
    theta = np.zeros(shape=(P, P))
    E = np.random.choice(int(P * (P - 1) / 2), k, replace=False)
    flatten_theta = np.zeros(int(P * (P - 1) / 2))
    flatten_theta[E] = (np.random.randint(2, size=k) - 0.5) * 2 * strength

    idx = 0
    for i in range(P):
        for j in range(i + 1, P):
            if idx in E:
                theta[i, j] = flatten_theta[idx]
                theta[j, i] = theta[i, j]
            idx += 1

    data = _skscope_experiment.ising_generator(n, theta, seed)
    return flatten_theta, IsingData(data[np.where(data[:, 0] > 0.5)[0],])


def loss_cvxpy(params, data):
    tmp = -2.0 * np.matmul(data.table[:, :, np.newaxis], data.table[:, np.newaxis, :])
    tmp[:, np.arange(data.p), np.arange(data.p)] = 0.0
    loss = 0.0
    for i in range(data.n):
        loss += data.freq[i] * cp.sum(
            cp.logistic(
                cp.sum(
                    cp.multiply(
                        params[data.index_translator],
                        tmp[i, :, :],
                    ),
                    axis=0,
                )
            )
        )
    return loss


def loss_jax(params, data):
    tmp = -2.0 * np.matmul(data.table[:, :, np.newaxis], data.table[:, np.newaxis, :])
    tmp[:, np.arange(data.p), np.arange(data.p)] = 0.0
    params_mat = params[data.index_translator]

    return jnp.dot(
        data.freq,
        jnp.sum(
            jnp.logaddexp(
                jnp.sum(
                    jnp.multiply(
                        params_mat[:, :],
                        tmp,
                    ),
                    axis=2,
                ),
                0,
            ),
            axis=1,
        ),
    )


def data_cpp_wrapper(data):
    return _skscope_experiment.IsingData(data.freq, data.table)


def loss_cpp(para, data):
    return _skscope_experiment.ising_loss(para, data)


def grad_cpp(para, data):
    return _skscope_experiment.ising_grad(para, data)


if __name__ == "__main__":
    true_params, data = data_generator(700, 45, 5, 708)

    # print(true_params)
    print(np.count_nonzero(true_params))
    # true_params = np.zeros_like(true_params)
    print(loss_jax(jnp.array(true_params), data))

    true_params_cvxpy = cp.Variable(len(true_params))
    true_params_cvxpy.value = true_params
    print(loss_cvxpy(true_params_cvxpy, data).value)
