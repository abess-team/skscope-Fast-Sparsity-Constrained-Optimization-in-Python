import parallel_experiment_util 
import random

def task(n, p, seed):
    random.seed(seed)
    disturbance = random.random() / 100
    results = []
    for method in ("+", "*"):
        result = {}
        result["method"] = method
        result["answer"] = n + p if method == "+" else n * p
        result["answer"] += disturbance
        results.append(result)
    return results


if __name__ == "__main__":
    in_key = ["n", "p", "seed"]
    out_key = ["method", "answer"]
    pe = parallel_experiment_util.ParallelExperiment(
        task, in_key, out_key, processes=2, name="example", memory_limit=0.5
    )
    
    if False:
        pe.check(n=10, p=2, seed=1)
    else:
        parameters = parallel_experiment_util.para_generator(
            {"n": [10], "p": [300,20]},
            {"n": [1, 2, 3], "p": [1, 2, 3]},
            repeat=10,
            seed=0,
        )
        
        pe.run(parameters)
        #pe.save()