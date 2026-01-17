import numpy as np


def safe_log1p_absences(x):
    """
    Transformation stable et picklable :
    - clip à 0 (sécurité)
    - log1p
    """
    return np.log1p(np.clip(x, 0, None))
