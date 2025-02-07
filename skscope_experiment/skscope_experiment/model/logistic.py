import jax.numpy as jnp
import numpy as np
import cvxpy as cp
from abess import make_glm_data
import _skscope_experiment

def data_generator(n, p, k, seed):
    coef = np.zeros(p)
    np.random.seed(seed)
    coef[np.random.choice(np.arange(p), k, replace=False)] = np.random.choice([1, -1], k) * 100
    # generate correlation matrix with exponential decay
    R = np.zeros((p, p))
    for i in range(p):
        for j in range(i, p):
            R[i, j] = 0.2 ** abs(i - j)
    R = R + R.T - np.identity(p)

    x = np.random.multivariate_normal(mean=np.zeros(p), cov=R, size=(n,))

    xbeta = np.matmul(x, coef)
    xbeta[xbeta > 30] = 30
    xbeta[xbeta < -30] = -30

    p = np.exp(xbeta) / (1 + np.exp(xbeta))
    y = np.random.binomial(1, p)

    return coef, (x, y)
    
def data_cpp_wrapper(data):
    return _skscope_experiment.RegressionData(data[0], data[1])

def loss_cpp(params, data):
    return _skscope_experiment.logistic_loss(params, data)

def grad_cpp(params, data):
    return _skscope_experiment.logistic_grad(params, data)

def loss_jax(params, data):
    Xbeta = data[0] @ params
    return jnp.sum(jnp.logaddexp(Xbeta, 0) - data[1] * Xbeta)

def loss_cvxpy(params, data):
    Xbeta = data[0] @ params
    return cp.sum(
        cp.logistic(Xbeta) - cp.multiply(data[1], Xbeta)
    )

if __name__ == "__main__":
     
    true_params, data = data_generator(20, 10, 5, 0)
    print(true_params)
    print(loss_jax(jnp.array(true_params), data))
    true_params_cvxpy = cp.Variable(len(true_params))
    true_params_cvxpy.value = true_params
    print(loss_cvxpy(true_params_cvxpy, data).value)
    """
    from jax import grad, jit 
    loss_grad = jit(grad(loss_jax))
    import time
    for n in range(100, 1001, 100):
        n *= 10
        t = 0.0
        for seed in range(10):
            true_params, data = data_generator(n, 500, 50, seed)
            loss_grad(true_params, data)
            for _ in range(10):
                params = np.zeros(500)
                params[np.random.choice(500, 50, replace=False)] = np.random.randn(50)
                start = time.time()
                grad(loss_jax)(params, data)
                t += (time.time() - start)
        print(n, t)
    """


