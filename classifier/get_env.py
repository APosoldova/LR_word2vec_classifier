import os

def get_env(key):
    try:
        return os.environ[key]
    except KeyError:
        raise RuntimeError(f"{key} must be present in the enviroment")

if __name__=="__main__":
    pyenv=get_ev("PY_ENV")
