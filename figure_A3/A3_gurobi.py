from itertools import product
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time
import parallel_experiment_util
from skscope_experiment.model import linear, non_linear_non_additive_example, trend_filtering_1d

TIME_LIMIT = 1000


def non_linear_task(n, p, k, seed):
    true_params, (features, response) = non_linear_non_additive_example.data_generator(n, p, k, seed)

    t1 = time.perf_counter()
    non_zero = k
    assert isinstance(non_zero, (int, np.integer))
    samples, dim = features.shape
    assert samples == response.shape[0]
    assert non_zero <= dim

    with gp.Env() as env, gp.Model(env=env) as regressor:
        regressor.setParam("output_flag", 0)
        regressor.setParam("threads", 1)
        regressor.setParam("node_file_start", 0.5)
        regressor.setParam("time_limit", TIME_LIMIT)
        # Append a column of ones to the feature matrix to account for the y-intercept
        X = features

        # Decision variables
        beta = regressor.addVars(dim, lb=0.0, name="beta")  # Weights
        # iszero[i] = 1 if beta[i] = 0
        iszero = regressor.addVars(dim, vtype=GRB.BINARY, name="iszero")

        # Objective Function (OF): minimize 1/2 * RSS using the fact that
        # if x* is a minimizer of f(x), it is also a minimizer of k*f(x) iff k > 0
        Quad = np.dot(X.T, X)
        lin = np.dot(response.T, X)
        obj = sum(0.5 * Quad[i, j] * beta[i] * beta[j] for i, j in product(range(dim), repeat=2))
        obj -= sum(lin[i] * beta[i] for i in range(dim))
        obj += 0.5 * np.dot(response, response)
        regressor.setObjective(obj, GRB.MINIMIZE)

        # Constraint sets
        for i in range(dim):
            # If iszero[i]=1, then beta[i] = 0
            regressor.addSOS(GRB.SOS_TYPE1, [beta[i], iszero[i]])
        regressor.addConstr(iszero.sum() == dim - non_zero)  # Budget constraint

        regressor.optimize()

        coeff = np.array([beta[i].X for i in range(dim)])
    t2 = time.perf_counter()

    return {
        "time": t2 - t1,
        "accuracy": len(set(np.nonzero(coeff)[0]) & set(np.nonzero(true_params)[0])) / k,
    }


def linear_task(n, p, k, seed):
    true_params, (features, response) = linear.data_generator(n, p, k, seed)

    t1 = time.perf_counter()
    non_zero = k
    assert isinstance(non_zero, (int, np.integer))
    samples, dim = features.shape
    assert samples == response.shape[0]
    assert non_zero <= dim

    with gp.Env() as env, gp.Model(env=env) as regressor:
        regressor.setParam("output_flag", 0)
        regressor.setParam("threads", 1)
        regressor.setParam("node_file_start", 0.5)
        regressor.setParam("time_limit", TIME_LIMIT)
        # Append a column of ones to the feature matrix to account for the y-intercept
        X = features

        # Decision variables
        beta = regressor.addVars(dim, lb=-GRB.INFINITY, name="beta")  # Weights
        # iszero[i] = 1 if beta[i] = 0
        iszero = regressor.addVars(dim, vtype=GRB.BINARY, name="iszero")

        # Objective Function (OF): minimize 1/2 * RSS using the fact that
        # if x* is a minimizer of f(x), it is also a minimizer of k*f(x) iff k > 0
        Quad = np.dot(X.T, X)
        lin = np.dot(response.T, X)
        obj = sum(0.5 * Quad[i, j] * beta[i] * beta[j] for i, j in product(range(dim), repeat=2))
        obj -= sum(lin[i] * beta[i] for i in range(dim))
        obj += 0.5 * np.dot(response, response)
        regressor.setObjective(obj, GRB.MINIMIZE)

        # Constraint sets
        for i in range(dim):
            # If iszero[i]=1, then beta[i] = 0
            regressor.addSOS(GRB.SOS_TYPE1, [beta[i], iszero[i]])
        regressor.addConstr(iszero.sum() == dim - non_zero)  # Budget constraint

        regressor.optimize()

        coeff = np.array([beta[i].X for i in range(dim)])
    t2 = time.perf_counter()

    return {
        "time": t2 - t1,
        "accuracy": len(set(np.nonzero(coeff)[0]) & set(np.nonzero(true_params)[0])) / k,
    }


def trend_filter_task(n, p, k, seed):
    true_params, ts = trend_filtering_1d.data_generator(n, p, k, seed)

    t1 = time.perf_counter()
    non_zero = k
    assert isinstance(non_zero, (int, np.integer))
    dim = p

    with gp.Env() as env, gp.Model(env=env) as regressor:
        regressor.setParam("output_flag", 0)
        regressor.setParam("threads", 1)
        regressor.setParam("node_file_start", 0.5)
        regressor.setParam("time_limit", TIME_LIMIT)

        X = np.zeros((p, p))
        X[np.tril_indices(p)] = 1.0
        # Decision variables
        beta = regressor.addVars(dim, lb=-GRB.INFINITY, name="beta")  # Weights
        # iszero[i] = 1 if beta[i] = 0
        iszero = regressor.addVars(dim, vtype=GRB.BINARY, name="iszero")

        # Objective Function (OF): minimize 1/2 * RSS using the fact that
        # if x* is a minimizer of f(x), it is also a minimizer of k*f(x) iff k > 0
        Quad = np.dot(X.T, X)
        lin = np.dot(ts.T, X)
        obj = sum(0.5 * Quad[i, j] * beta[i] * beta[j] for i, j in product(range(dim), repeat=2))
        obj -= sum(lin[i] * beta[i] for i in range(dim))
        obj += 0.5 * np.dot(ts, ts)
        regressor.setObjective(obj, GRB.MINIMIZE)

        # Constraint sets
        for i in range(dim):
            # If iszero[i]=1, then beta[i] = 0
            regressor.addSOS(GRB.SOS_TYPE1, [beta[i], iszero[i]])
        regressor.addConstr(iszero.sum() == dim - non_zero)  # Budget constraint

        regressor.optimize()

        coeff = np.array([beta[i].X for i in range(dim)])
    t2 = time.perf_counter()

    return {
        "time": t2 - t1,
        "accuracy": len(set(np.nonzero(coeff)[0]) & set(np.nonzero(true_params)[0])) / k,
    }


def task(model, n, p, k, seed):
    if model == "linear":
        return linear_task(n, p, k, seed)
    elif model == "non_linear":
        return non_linear_task(n, p, k, seed)
    elif model == "trend_filter":
        return trend_filter_task(n, p, k, seed)
    else:
        raise ValueError("model is invalid!")


if __name__ == "__main__":
    in_keys = ["model", "n", "p", "k", "seed"]
    out_keys = [
        "time",
        "accuracy",
    ]

    experiment = parallel_experiment_util.ParallelExperiment(
        task=task,
        in_keys=in_keys,
        out_keys=out_keys,
        processes=10,
        name="A3_gurobi",
        memory_limit=0.9,
    )

    if False:
        experiment.check(model="trend_filter", n=600, p=600, k=50, seed=1)
    else:
        parameters = parallel_experiment_util.para_generator(
            {
                "model": ["trend_filter"],
                "n": [600],
                "p": [600],
                "k": [50],
            },
            {
                "model": ["non_linear"],
                "n": [600],
                "p": [50],
                "k": [10],
            },
            {
                "model": ["linear"],
                "n": [600],
                "p": [500],
                "k": [50],
            },
            repeat=100,
            seed=0,
        )

        experiment.run(parameters)
        experiment.save()
