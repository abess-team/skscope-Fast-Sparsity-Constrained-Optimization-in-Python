from itertools import product
import numpy as np
from functools import wraps
import time

def product_dict(**kwargs):
    """
    Usage:
        n = [500,1000,2000]
        p = [1000,2000]

        def test(n=1,p=1):
            print(n,p)

        for instance in product_dict(n=n,p=p):
            test(**instance)
    """
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in product(*vals):
        yield dict(zip(keys, instance))

def merge_dict(list_of_dict1, list_of_dict2):
    """
    Usage:
        d1 = [{'n': 1, 'm':2},{'n': 11, 'm': 12}]
        d2 = [{'a':0.9,'fpr': 0.1}, {'a':0.92,'fpr': 0.14}]
        r = []
        r.extend(merge_dict(d1, d2))
        print(r)
    """
    for d in zip(list_of_dict1, list_of_dict2):
        d[0].update(d[1])
        yield d[0]

def del_duplicate(*lists_of_dict):
    """
    >>>list(del_duplicate([{"a": 1}, {"b": 2}, {"a": 1}], [{"a": 2}, {"a": 1}]))
    >>>[{'b': 2}, {'a': 1}, {'a': 2}]
    """
    for tuple_dict in set(
        [tuple(dict.items()) for list_of_dict in lists_of_dict for dict in list_of_dict]
    ):
        yield dict(tuple_dict)

def accuracy(model_coef, data_coef):
    """
    use fo variables selection
    """
    model_coef = set(np.nonzero(model_coef)[0])
    data_coef = set(np.nonzero(data_coef)[0])
    return len(model_coef & data_coef) / len(data_coef)

def FDR(model_coef, data_coef):
    """
    use fo variables selection
    """
    model_coef = set(np.nonzero(model_coef)[0])
    data_coef = set(np.nonzero(data_coef)[0])
    return len(model_coef - data_coef) / len(model_coef) if len(model_coef) > 0 else 0

def para_generator(*args, repeat=1, seed=None):
    """
    Usage:
        para_generator(
            {"n": [i * 100 + 100 for i in range(5)], "p": [500], "k": [50]},
            {"n": [500], "p": [i * 100 + 100 for i in range(5)], "k": [50]},
            repeat=3,
            seed=1234
        )
    """
    # if repeat is not iterable, make it iterable
    if not hasattr(repeat, "__iter__"):
        repeat = [repeat] * len(args)

    for group_of_para, iter in zip(args,repeat):
        for para in product_dict(**group_of_para):
            for i in range(iter):
                if seed is not None:
                    para['seed'] = seed
                    seed += 1
                yield para.copy()

def timethis(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.process_time()
        r = func(*args, **kwargs)
        end = time.process_time()
        return r, end - start

    return wrapper

if __name__ == "__main__":
    print([para for para in para_generator(
            {"n": [i * 100 + 100 for i in range(2)], "p": [500], "k": [50]},
            {"n": [500], "p": [i * 100 + 100 for i in range(2)], "k": [50]},
            repeat=[2,3],
            seed=1234
    )])
   
