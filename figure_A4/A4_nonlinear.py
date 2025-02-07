import parallel_experiment_util
from skscope import (
    ScopeSolver,
    GraspSolver,
    IHTSolver,
    HTPSolver,
    FobaSolver,
    OMPSolver,
)
from skscope.utilities import LinearSIC, SIC, AIC, BIC, EBIC
import time
import numpy as np
from skscope_experiment import model as Model
import cvxpy as cp

model_dict = {
    "ising": Model.ising,
    "non_linear": Model.non_linear_non_additive_example,
    "robust_ESL": Model.robust_ESL,
}


def task(model: str, sample_size, dim, sparsity_level, seed):
    results = []
    true_params, data = model_dict[model].data_generator(sample_size, dim, sparsity_level, seed)
    true_support_set = set(np.nonzero(true_params)[0])
    loss_jax = lambda x: model_dict[model].loss_jax(x, data)
    ic_method = SIC if model != "robust_ESL" else AIC

    for method, solver in {
        "SCOPE": ScopeSolver(dim, sparsity=range(sparsity_level * 2), sample_size=sample_size, ic_method=ic_method),
        "GraSP": GraspSolver(dim, sparsity=range(sparsity_level * 2), sample_size=sample_size, ic_method=ic_method),
        "FoBa": FobaSolver(dim, sparsity=range(sparsity_level * 2), sample_size=sample_size, ic_method=ic_method),
        "OMP": OMPSolver(dim, sparsity=range(sparsity_level * 2), sample_size=sample_size, ic_method=ic_method),
    }.items():
        t1 = time.time()
        solver.solve(loss_jax, jit=True)
        t2 = time.time()
        results.append(
            {
                "method": method,
                "time": t2 - t1,
                "accuracy": len(set(solver.get_support()) & true_support_set) / sparsity_level,
                "support_size": len(solver.get_support()),
            }
        )

    # IHT, HTP
    for method, solver in {
        "IHT": IHTSolver(dim, sparsity=range(sparsity_level * 2), sample_size=sample_size, ic_method=ic_method),
        "HTP": HTPSolver(dim, sparsity=range(sparsity_level * 2), sample_size=sample_size, ic_method=ic_method),
    }.items():
        t1 = time.time()
        best_loss = np.inf
        for step_size in [1.0, 0.1, 0.01, 0.001, 0.0001]:
            solver.set_params(step_size=step_size)
            try:
                solver.solve(loss_jax, jit=True)
            except:
                continue
            loss = solver.objective_value
            if loss < best_loss:
                best_loss = loss
                result = {
                    "method": method,
                    "time": 0,
                    "accuracy": len(set(solver.get_support()) & true_support_set) / sparsity_level,
                    "support_size": len(solver.get_support()),
                }
        result["time"] = time.time() - t1
        results.append(result)

    # CVXPY
    if model == "non_linear":
        x = cp.Variable(dim)

        def object_fn(x, lambd):
            return model_dict[model].loss_cvxpy(x, data) + lambd * cp.norm1(x)

        lambd = cp.Parameter(nonneg=True)
        problem = cp.Problem(cp.Minimize(object_fn(x, lambd)), constraints=model_dict[model].cvxpy_constraints(x))
        lambd_lowwer = 0.0
        lambd.value = 100.0
        best_ic = np.inf
        start = time.time()
        for target_sparsity in range(sparsity_level * 2 - 1, -1, -1):
            try:
                problem.solve()
            except:
                return results
            params = x.value
            support_set = set(np.where(abs(params) > 1e-2)[0])  # set(np.nonzero(params)[0])
            ht_params = np.zeros_like(params)
            ht_params[list(support_set)] = params[list(support_set)]

            ic_value = 2 * loss_jax(ht_params) + len(support_set) * 2  # np.log(sample_size)

            print(
                lambd.value,
                loss_jax(ht_params),
                ic_value,
                len(support_set & true_support_set) / sparsity_level,
                len(support_set),
                target_sparsity,
            )

            if ic_value < best_ic:
                best_ic = ic_value
                result = {
                    "method": "LASSO",
                    "time": 0,
                    "accuracy": len(support_set & true_support_set) / sparsity_level,
                    "support_size": len(support_set),
                }
            if len(support_set) > target_sparsity:
                lambd_lowwer = lambd.value
                lambd.value = 2 * lambd.value
            elif len(support_set) < target_sparsity:
                lambd.value = (lambd_lowwer + lambd.value) / 2
        result["time"] = time.time() - start
        results.append(result)

    return results


if __name__ == "__main__":
    experiment = parallel_experiment_util.ParallelExperiment(
        task=task,
        in_keys=["model", "sample_size", "dim", "sparsity_level", "seed"],
        out_keys=["method", "time", "accuracy", "support_size"],
        processes=10,
        name="A4_nonlinear",
        memory_limit=0.9,
    )
    if False:
        # experiment.check(model="multitask", sample_size=600, dim=500, sparsity_level=50, seed=1)
        experiment.check(model="non_linear", sample_size=600, dim=50, sparsity_level=10, seed=1)
        # experiment.check(model="linear", sample_size=600, dim=500, sparsity_level=50, seed=294)
        # experiment.check(model="logistic", sample_size=600, dim=500, sparsity_level=50, seed=100)
        # experiment.check(model="ising", sample_size=600, dim=190, sparsity_level=40, seed=200)
        # experiment.check(model="trend_filter", sample_size=600, dim=600, sparsity_level=50, seed=90)
        # experiment.check(model="robust_ESL", sample_size=600, dim=500, sparsity_level=50, seed=900)
    else:
        parameters = parallel_experiment_util.para_generator(
            {
                "model": ["non_linear"],
                "sample_size": [600],
                "dim": [50],
                "sparsity_level": [10],
            },
            {
                "model": ["ising"],
                "sample_size": [600],
                "dim": [190],
                "sparsity_level": [40],
            },
            {
                "model": ["robust_ESL"],
                "sample_size": [600],
                "dim": [500],
                "sparsity_level": [50],
            },
            repeat=100,
            seed=0,
        )

        experiment.run(parameters)
        experiment.save()
