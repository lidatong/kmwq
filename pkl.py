import pickle
from functools import wraps


def pkl_dump(obj, path):
    with open(path, 'wb') as fp:
        pickle.dump(obj, fp)


def pkl_load(path):
    with open(path, 'rb') as fp:
        return pickle.load(fp)


def pkl(path, dump=None, load=None):
    dump = pkl_dump if dump is None else dump
    load = pkl_load if load is None else load

    def pkl_f(f):
        @wraps(f)
        def cached_f(*args, **kwargs):
            try:
                res = load(path)
            except FileNotFoundError:
                print(f"Saved results of {f.__name__} not found. Computing...")
                res = f(*args, **kwargs)
                dump(res, path)
            else:
                print(f"Retrieving saved results of {f.__name__}.")
            return res

        return cached_f

    return pkl_f
