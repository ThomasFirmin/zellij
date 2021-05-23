import numpy as np



def select(name):

    if name=="henon_map":
        return henon_map
    elif name=="inverted_henon":
        return inverted_henon
    elif name=="logistic_map":
        return henon_map
    elif name=="tent_map":
        return henon_map
    elif name=="kent_map":
        return kent_map
    elif name=="random_map":
        return random_map
    elif name=="mixed":
        return mixed
    else:
        print("Invalid map name")

# Define chaotic variable according to Henon Map
def henon_map(n_vectors: int, n_param: int, a:float=1.4020560, b:float=0.305620406) -> np.ndarray:

    # Initialization
    x = np.zeros([n_vectors,n_param])
    y = np.zeros([n_vectors,n_param])

    x[0,:] = np.random.random(n_param)

    for i in range(1,n_vectors):

        # x_{k+1} = a.(1-x_{k}^2) + b.y_{k}
        x[i,:] = 1-a*np.square(x[i-1,:])+y[i-1,:]

        # y_{k+1} = x_{k}
        y[i,:] = b*x[i-1,:]

    # Min_{n_param}(y_{n_param,n_vectors})
    alpha = np.amin(y, axis=0)

    # Max_{n_param}(y_{n_param,n_vectors})
    beta = np.amax(y, axis=0)

    return (y-alpha)/(beta-alpha)

def inverted_henon(n_vectors: int, n_param: int, a:float=1.5, b:float=0.2) -> np.ndarray:
    return henon_map(n_vectors,n_param,a,b)*1.99-1

def logistic_map(n_vectors: int, n_param: int, mu:float=3.57) -> np.ndarray:
    x = np.zeros([n_vectors,n_param])
    x[0,:]=np.random.random(n_param)

    for i in range(1,n_vectors):
        x[i,:] = mu*x[i-1,:]*(1-x[i-1,:])

    return x

def kent_map(n_vectors: int, n_param: int, beta:float=0.8) -> np.ndarray:
    x = np.zeros([n_vectors,n_param])
    x[0,:]=np.random.random(n_param)

    for i in range(1,n_vectors):
        x[i,:] = np.where(x[i-1,:]<beta,x[i-1,:]/beta,(1-x[i-1,:])/(1-beta))

    return x

def tent_map(n_vectors: int, n_param: int, mu:float=0.8) -> np.ndarray:
    x = np.zeros([n_vectors,n_param])

    for i in range(1,n_vectors):
        x[i,:] = mu*(1-2*np.absolute((x[i-1,:]-0.5)))

    return x

def mixed(n_vectors: int, n_param: int) -> np.ndarray:
    x = np.zeros([n_vectors,n_param])

    y = henon_map(n_vectors, n_param) + inverted_henon(n_vectors, n_param)

    # Min_{n_param}(y_{n_param,n_vectors})
    alpha = np.amin(y, axis=0)

    # Max_{n_param}(y_{n_param,n_vectors})
    beta = np.amax(y, axis=0)

    return (y-alpha)/(beta-alpha)


def random_map(n_vectors: int, n_param: int) -> np.ndarray:
    return np.random.random((n_vectors,n_param))
