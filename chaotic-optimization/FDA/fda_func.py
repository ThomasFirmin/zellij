import numpy as np
from hypersphere import Hypersphere

import tornado_utils.chaos_map as cmap

# Lazy Hypervolume Search
def LHS(H,loss_func):
    loss_call = 0

    for i in range(H.dim):
        inf = np.copy(H.center)
        sup = np.copy(H.center)

        inf[i] = np.max([H.center[i]-H.radius[i]/np.sqrt(H.dim),H.lo_bounds[i]])
        sup[i] = np.min([H.center[i]+H.radius[i]/np.sqrt(H.dim),H.up_bounds[i]])

        score1 = loss_func(inf)
        score2 = loss_func(sup)

        loss_call += 2
        H.add_point(score1, inf,"blue")
        H.add_point(score2, sup,"blue")

    return loss_call

def ILS(H,loss_func,red_rate=0.5,precision=1e-5):

    loss_call = 0

    scores = [0]*3
    solutions = [np.copy(H.center),np.copy(H.center),np.copy(H.center)]
    scores[0] = loss_func(solutions[0])
    H.add_point(scores[0], solutions[0], "cyan")

    loss_call += 1

    step = np.max(H.radius)

    while step > precision:

        for i in range(H.dim):

            walk = solutions[0][i] + step
            db = np.min([H.up_bounds[i],walk])
            solutions[1][i] = db

            scores[1] = loss_func(solutions[1])
            H.add_point(scores[1], solutions[1], "cyan")

            loss_call += 1

            walk = solutions[0][i] - step
            db = np.max([H.lo_bounds[i],walk])
            solutions[2][i] = db

            scores[2] = loss_func(solutions[2])
            H.add_point(scores[2], solutions[2], "cyan")

            loss_call += 1

            min_index = np.argmin(scores)
            solutions = [np.copy(solutions[min_index]),np.copy(solutions[min_index]),np.copy(solutions[min_index])]

            H.add_point(scores[min_index], solutions[min_index], "red")

        step = red_rate * step

    return loss_call

def CGS(H,loss_func,n_level_cgs=10,chaos_map_name="henon_map"):

    center = H.center
    radius = H.radius
    general_lo_bounds = H.lo_bounds
    general_up_bounds = H.up_bounds

    lo_bounds = np.maximum(center-radius,general_lo_bounds)
    up_bounds = np.minimum(center+radius,general_up_bounds)

    dim = len(center)

    map_size = n_level_cgs

    chaos_map = cmap.select(chaos_map_name)
    chaos_variables = chaos_map(map_size,dim)
    one_m_chaos_variables = 1-chaos_variables

    # For each level of chaos
    points = np.zeros((4*n_level_cgs,dim))

    up_m_lo = np.subtract(up_bounds,lo_bounds)

    n_points = 0
    loss_call = 0
    for l in range(n_level_cgs):

        # Select chaotic_variables among the choatic map
        y = chaos_variables[l]
        # Apply 3 transformations on the selected chaotic variables
        r_mul_y = np.multiply(up_m_lo,y)

        xx = [np.add(lo_bounds,r_mul_y),np.subtract(up_bounds,r_mul_y)]

        # Randomly select a parameter index of a solution
        d=np.random.randint(dim)

        # for each transformation of the chaotic variable
        sym = np.matrix([xx[0],xx[1],xx[0],xx[1]])
        sym[2,d] = xx[1][d]
        sym[3,d] = xx[0][d]
        points[n_points: n_points+4] = sym
        n_points += 4

    for p in points:
        score = loss_func(p)
        loss_call += 1
        add_p(H, score, p, "blue")

    return loss_call

def CLS(H,loss_func,n_level_cls=5,n_repetition=20,n_symetric_p=8,chaos_map_name="henon_map",red_rate=0.5):

    center = H.center
    radius = H.radius
    lo_bounds = H.lo_bounds
    up_bounds = H.up_bounds

    dim = len(center)

    trigo_val = 2*np.pi/n_symetric_p
    trigo = [np.zeros(n_symetric_p),np.zeros(n_symetric_p)]

    for i in range(1,n_symetric_p+1):
        # Initialize trigonometric part of symetric variables (CLS & CFS)
        trigo[0][i-1] = np.cos(trigo_val*i)
        trigo[1][i-1] = np.sin(trigo_val*i)

    solution = np.copy(center)
    min_score = loss_func(solution)

    map_size = n_level_cls

    loss_calls = 0

    for k in range(n_repetition):

        chaos_map = cmap.select(chaos_map_name)
        chaos_variables = chaos_map(map_size,dim)
        one_m_chaos_variables = 1-chaos_variables

        up_m_lo = np.subtract(up_bounds,lo_bounds)

        # Limits of the search area, if parameter greater than center, then = 1 else = -1, used to avoid overflow
        db = np.minimum(up_bounds-solution,solution-lo_bounds)

        # Local search area radius
        Rl = radius*red_rate

        center_m_solution = center - solution
        points = np.zeros((2*n_level_cls*n_symetric_p,dim))

        n_points = 0
        # for each level of chaos
        for l in range(n_level_cls):

            # Decomposition vector
            d = np.random.randint(dim)

            # zoom speed
            gamma = 1/(10**(2*red_rate*l)*(l+1))

            # for each parameter of a solution determine the improved radius
            xx = np.minimum(gamma*Rl,db)

            # Compute both chaotic variable of the polygonal model thanks to a chaotic map
            xv = [np.multiply(xx,chaos_variables[l]),np.multiply(xx,one_m_chaos_variables[l])]

            # For both chaotic variable
            for x in xv :
                xi = np.outer(trigo[1],x)
                xi[:,d] = x[d]*trigo[0]
                xt = solution + xi

                points[n_points: n_points+n_symetric_p] = xt
                n_points += n_symetric_p

        for p in points:

            score = loss_func(p)

            if score < min_score:
                min_score = score
                solution = np.copy(p)

            loss_calls += 1

            add_p(H, score, p, "cyan")

    return loss_calls

##################################################################################################################################################################################################
#_____________________________________________________________________________________Distributed functions_____________________________________________________________________________________#
##################################################################################################################################################################################################

def LHS_mpi(essential_infos,scores,last_state,initialize = False):

    if initialize:
        return -1,-1
    else:
        center = essential_infos[0]
        radius = essential_infos[1]
        lo_bounds = essential_infos[2]
        up_bounds = essential_infos[3]
        dim = len(center)


        points = []
        for i in range(dim):
            inf = np.copy(center)
            sup = np.copy(center)

            inf[i] = np.max([center[i]-radius[i]/np.sqrt(dim),lo_bounds[i]])
            sup[i] = np.min([center[i]+radius[i]/np.sqrt(dim),up_bounds[i]])

            points.append(np.copy(inf))
            points.append(np.copy(sup))

        finished = 1
        actual_state = None

        return points,finished,actual_state

def ILS_mpi(essential_infos,new_scores,last_state,initialize = False ,red_rate=0.5,precision=1e-5):

    center = essential_infos[0]
    radius = essential_infos[1]
    lo_bounds = essential_infos[2]
    up_bounds = essential_infos[3]
    dim = len(center)

    if initialize:
        step = np.max(radius)
        last_state = [[np.copy(center),np.copy(center),np.copy(center)],step,0]
        return -1,last_state
    else:

        scores = new_scores
        solutions = last_state[0]

        if scores != -1:

            min_index = np.argmin(scores)
            solutions = [np.copy(solutions[min_index]),np.copy(solutions[min_index]),np.copy(solutions[min_index])]

        step = last_state[1]
        i = last_state[2]

        walk = solutions[0][i] + step
        db = np.min([up_bounds[i],walk])
        solutions[1][i] = db

        walk = solutions[0][i] - step
        db = np.max([lo_bounds[i],walk])
        solutions[2][i] = db

        step = red_rate * step

        if i < dim-1:
            i += 1
        else:
            i = 0
            step = red_rate * step

        if step < precision:
            finished = 1
        else:
            finished = 0

        last_state = [solutions,step,i]


        return solutions,finished,last_state

def CGS_mpi(essential_infos,new_scores,last_state,initialize = False,n_level_cgs=5,chaos_map_name="henon_map"):

    if initialize:
        return -1,-1

    else:

        center = essential_infos[0]
        radius = essential_infos[1]
        general_lo_bounds = essential_infos[2]
        general_up_bounds = essential_infos[3]

        lo_bounds = np.maximum(center-radius,general_lo_bounds)
        up_bounds = np.minimum(center+radius,general_up_bounds)

        dim = len(center)

        map_size = n_level_cgs

        chaos_map = cmap.select(chaos_map_name)
        chaos_variables = chaos_map(map_size,dim)
        one_m_chaos_variables = 1-chaos_variables

        # For each level of chaos
        points = np.zeros((4*n_level_cgs,dim))

        up_m_lo = np.subtract(up_bounds,lo_bounds)

        n_points = 0

        for l in range(n_level_cgs):

            # Select chaotic_variables among the choatic map
            y = chaos_variables[l]
            # Apply 3 transformations on the selected chaotic variables
            r_mul_y = np.multiply(up_m_lo,y)

            xx = [np.add(lo_bounds,r_mul_y),np.subtract(up_bounds,r_mul_y)]

            # Randomly select a parameter index of a solution
            d=np.random.randint(dim)

            # for each transformation of the chaotic variable
            sym = np.matrix([xx[0],xx[1],xx[0],xx[1]])
            sym[2,d] = xx[1][d]
            sym[3,d] = xx[0][d]
            points[n_points: n_points+4] = sym
            n_points += 4

        actual_state = None

        return points.tolist(),1,actual_state

def CLS_mpi(essential_infos,new_scores,last_state,initialize = False,n_repetition=5,n_level_cls=4,n_symetric_p=16,chaos_map_name="henon_map",red_rate=0.5):

    center = essential_infos[0]
    radius = essential_infos[1]
    lo_bounds = essential_infos[2]
    up_bounds = essential_infos[3]

    dim = len(center)

    if initialize:

        trigo_val = 2*np.pi/n_symetric_p
        H = [np.zeros(n_symetric_p),np.zeros(n_symetric_p)]

        for i in range(1,n_symetric_p+1):
            # Initialize trigonometric part of symetric variables (CLS & CFS)
            H[0][i-1] = np.cos(trigo_val*i)
            H[1][i-1] = np.sin(trigo_val*i)

        map_size = n_level_cls*n_repetition
        chaos_map = cmap.select(chaos_map_name)
        chaos_variables = chaos_map(map_size,dim)
        one_m_chaos_variables = 1-chaos_variables

        last_state = [[],np.copy(center),0,H,float("inf"),chaos_variables,one_m_chaos_variables]

        return -1,last_state

    else:

        k = last_state[2]
        H = last_state[3]
        min_score = last_state[4]
        chaos_variables = last_state[5]
        one_m_chaos_variables = last_state[6]

        shift = (k-1)*n_level_cls

        if new_scores != -1:

            solutions = last_state[0]

            idx = np.argmin(new_scores)

            if new_scores[idx] < min_score:
                min_score = new_scores[idx]
                last_state[4] = new_scores[idx]
                solution = solutions[idx]
                last_state[1] = solution
            else:
                solution = last_state[1]

        else:
            solution = last_state[1]



        up_m_lo = np.subtract(up_bounds,lo_bounds)

        # Limits of the search area, if parameter greater than center, then = 1 else = -1, used to avoid overflow
        db = np.minimum(up_bounds-solution,solution-lo_bounds)

        # Local search area radius
        Rl = radius*red_rate

        center_m_solution = center - solution
        points = np.zeros((2*n_level_cls*n_symetric_p,dim))

        n_points = 0

        # for each level of chaos
        for l in range(n_level_cls):

            # Decomposition vector
            d = np.random.randint(dim)

            # zoom speed
            gamma = 1/(10**(2*red_rate*l)*(l+1))

            # for each parameter of a solution determine the improved radius
            xx = np.minimum(gamma*Rl,db)

            # Compute both chaotic variable of the polygonal model thanks to a chaotic map
            xv = [np.multiply(xx,chaos_variables[shift]),np.multiply(xx,one_m_chaos_variables[shift])]

            # For both chaotic variable
            for x in xv :
                xi = np.outer(H[1],x)
                xi[:,d] = x[d]*H[0]
                xt = solution + xi

                points[n_points: n_points+n_symetric_p] = xt
                n_points += n_symetric_p

        k += 1
        last_state[0] = np.copy(points)
        last_state[1] = np.copy(solution)
        last_state[2] = k
        last_state[3] = H

        if k > n_repetition:
            finished = 1
        else:
            finished = 0

        return points.tolist(),finished,last_state
