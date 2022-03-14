import numpy as np


def minimum(H, *arg, **kwargs):
    """best(H, *arg, **kwargs)

    Parameters
    ----------
    H : Fractal
        Fractal on which to compute the heuristic value.

    Returns
    -------
    _ : float
        Minimum score found inside the fractal
    """

    return H.min_score


def median(H, *args, **kwargs):
    """median(H, *args, **kwargs)

    Parameters
    ----------
    H : Fractal
        Fractal on which to compute the heuristic value.

    Returns
    -------
    _ : float
        Median of all scores computed inside the fractal
    """

    return np.median(H.all_scores)


def mean(H, *args, **kwargs):
    """mean(H,best_ind,best_sc)

    Parameters
    ----------
    H : Fractal
        Fractal on which to compute the heuristic value.

    Returns
    -------
    _ : float
        Mean score of all scores computed inside the fractal
    """

    return np.mean(H.all_scores)


def std(H, *args, **kwargs):

    """_std(H, *args, **kwargs)

    Parameters
    ----------
    H : Fractal
        Fractal on which to compute the heuristic value.

    Returns
    -------
    _ : float
        Standard deviation of all scores computed inside the fractal
    """

    return np.std(H.all_scores)


def dttcb(H, *args, **kwargs):

    """dttcb(H,best_ind,best_sc)

    Parameters
    ----------
    H : Fractal
        Fractal on which to compute the heuristic value.

    Returns
    -------
    _ : float
        Distance to the best found solution so far to the best found solution computed inside the fractal
    """

    best_ind = args[0]

    return np.min(np.array(H.all_scores) / np.linalg.norm(np.array(H.solutions) - np.array(best_ind), axis=1))


def belief(H, gamma=0.5, *args, **kwargs):

    """belief(H,best_ind,best_sc)

    Parameters
    ----------
    H : Fractal
        Fractal on which to compute the heuristic value.

    gamma : float, default=0.5
        Influences the contribution of the father's best score and the mean of all scores inside the fractal.
        Higher gamma means higher contribution of the fractal's father.

    Returns
    -------
    _ : float
        Belief computed according to the influences of the father's score and the mean of all computed solutions inside the fractal
    """

    best_sc = args[1]

    if type(H.father.father) == str:
        H.father.score = 0

    ratio = np.array(H.all_scores) / best_sc
    return -(gamma * H.father.score + (1 - gamma) * np.mean(ratio * np.exp(1 - ratio)))
