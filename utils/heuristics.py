import numpy as np

heuristic_list = {"best":_best,"median":_median,"mean":_mean,"std":_std,"dttcb":_dttcb,"belief":_belief}

def _best(H,best_ind,best_sc):
    return H.min_score

def _median(H,best_ind,best_sc):
    return np.median(H.all_scores)

def _mean(H,best_ind,best_sc):
    return np.mean(H.all_scores)

def _std(H,best_ind,best_sc):
    return np.std(H.all_scores)

def _dttcb(H,best_ind,best_sc):
    return H.min_score/np.linalg.norm(np.array(H.best_sol_c)-np.array(best_ind))

def _belief(H,best_ind,best_sc,gamma=0.5):

    if type(H.father.father) == str:
        H.father.score = 0

    ratio = np.array(H.all_scores)/best_sc
    return -(gamma*H.father.score + (1-gamma)*np.mean(ratio*np.exp(1-ratio)))
