import numpy as np

def _best(H,best_ind,best_sc):
    """_best(H,best_ind,best_sc)

    Parameters
    ----------
    H : Fractal
        Fractal on which to compute the heuristic value.
    best_ind : list
        Best solution found so far.
    best_sc : score
        Best score found so far.

    Returns
    -------
    _ : float
        Minimum score found inside the fractal
    """

    return H.min_score

def _median(H,best_ind,best_sc):
    """_median(H,best_ind,best_sc)

    Parameters
    ----------
    H : Fractal
        Fractal on which to compute the heuristic value.
    best_ind : list
        Best solution found so far.
    best_sc : score
        Best score found so far.

    Returns
    -------
    _ : float
        Median of all scores computed inside the fractal
    """

    return np.median(H.all_scores)

def _mean(H,best_ind,best_sc):
    """_mean(H,best_ind,best_sc)

    Parameters
    ----------
    H : Fractal
        Fractal on which to compute the heuristic value.
    best_ind : list
        Best solution found so far.
    best_sc : score
        Best score found so far.

    Returns
    -------
    _ : float
        Mean score of all scores computed inside the fractal
    """

    return np.mean(H.all_scores)

def _std(H,best_ind,best_sc):

    """_std(H,best_ind,best_sc)

    Parameters
    ----------
    H : Fractal
        Fractal on which to compute the heuristic value.
    best_ind : list
        Best solution found so far.
    best_sc : score
        Best score found so far.

    Returns
    -------
    _ : float
        Standard deviation of all scores computed inside the fractal
    """

    return np.std(H.all_scores)

def _dttcb(H,best_ind,best_sc):

    """_dttcb(H,best_ind,best_sc)

    Parameters
    ----------
    H : Fractal
        Fractal on which to compute the heuristic value.
    best_ind : list
        Best solution found so far.
    best_sc : score
        Best score found so far.

    Returns
    -------
    _ : float
        Distance to the best found solution so far to the best found solution computed inside the fractal
    """

    return H.min_score/np.linalg.norm(np.array(H.best_sol_c)-np.array(best_ind))

def _belief(H,best_ind,best_sc,gamma=0.5):

    """_belief(H,best_ind,best_sc)

    Parameters
    ----------
    H : Fractal
        Fractal on which to compute the heuristic value.
    best_ind : list
        Best solution found so far.
    best_sc : score
        Best score found so far.
    gamma : float, default=0.5
        Influences the contribution of the father's best score and the mean of all scores inside the fractal.
        Higher gamma means higher contribution of the fractal's father.

    Returns
    -------
    _ : float
        Belief computed according to the influences of the father's score and the mean of all computed solutions inside the fractal
    """

    if type(H.father.father) == str:
        H.father.score = 0

    ratio = np.array(H.all_scores)/best_sc
    return -(gamma*H.father.score + (1-gamma)*np.mean(ratio*np.exp(1-ratio)))

heuristic_list = {"best":_best,"median":_median,"mean":_mean,"std":_std,"dttcb":_dttcb,"belief":_belief}
