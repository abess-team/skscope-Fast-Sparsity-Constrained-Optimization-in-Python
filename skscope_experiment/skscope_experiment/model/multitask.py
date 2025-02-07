import jax.numpy as jnp
import numpy as np
import cvxpy as cp
from sklearn.datasets import make_regression
import _skscope_experiment

def data_generator(n, p, k, seed):
    X, y, coef = make_regression(n_samples=n, n_features=p, n_informative=k, n_targets=3, coef=True, random_state=seed)
    return coef.flatten(), (X, y)

def loss_jax(params, data):
    p = data[0].shape[1]
    m = data[1].shape[1]
    return jnp.sum(jnp.square(data[1] - data[0] @ params.reshape((p, m))))

def loss_cvxpy(params, data):
    return cp.sum_squares(data[1] - data[0] @ params) 

def data_cpp_wrapper(data):
    return _skscope_experiment.RegressionData(data[0], data[1])

def loss_cpp(params, data):
    return _skscope_experiment.multitask_loss(params, data)

def grad_cpp(params, data):
    return _skscope_experiment.multitask_grad(params, data)

if __name__ == "__main__":
    true_params, data = data_generator(20, 5, 3, 0)
    print(true_params)
    true_params *= 1.5
    print(loss_jax(jnp.array(true_params), data))
    true_params_cvxpy = cp.Variable(len(true_params))
    true_params_cvxpy.value = true_params
    print(loss_cvxpy(true_params_cvxpy, data).value)
    print(loss_cpp(true_params, data_cpp_wrapper(data)))

    import jax
    print(jax.grad(loss_jax)(jnp.array(true_params), data))
    print(grad_cpp(true_params, data_cpp_wrapper(data)))
