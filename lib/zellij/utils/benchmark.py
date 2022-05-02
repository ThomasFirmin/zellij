import numpy as np


def shifted_cigar_bent(y,bias=100):
    y = np.array(y)
    shifted = y+1/len(y)
    return shifted[0]**2+(10**6)*np.sum(np.square(shifted[1:]))+bias

def shifted_rastrigin(y,bias=200):
    y = np.array(y)
    shifted = y+1/len(y)
    return np.sum(np.add(np.subtract(np.square(shifted),10*np.cos(2*np.pi*shifted)),10))+bias

def himmelblau(y):
    y = np.array(y)
    return np.sum(y**4 -16*y**2 + 5*y) * (1/len(y))

def alpine(y):
    y = np.array(y)
    return 1/len(y)*np.sum(np.absolute(y*np.sin(y)+0.1*y))

def ackley(y,bias=1100):
    y = np.array(y)
    return -20*np.exp(-0.2*np.sqrt(1/len(y)*np.sum(y**2)))-np.exp(1/len(y)*np.sum(np.cos(np.pi*2*y)))+20+np.exp(1)+bias

def happycat(y,bias=100):
    y = np.array(y)
    return np.absolute(np.sum(np.square(y)-len(y)))**(1/4)+0.5/len(y)*(np.sum(np.square(y))-np.sum(y))+0.5+bias

def shifted_levy(y,bias=500):
    y = np.array(y)
    return np.sin(np.pi*y[0])**2+np.sum((y[:-1]-1)**2*(1+10*np.sin(np.pi*y[:-1]+1)**2))+(y[-1]-1)**2*(1+np.sin(2*np.pi*y[-1])**2)+bias

def brown(y):
    y = np.array(y)
    return np.sum((y[:-1]**2)**(y[1:]**2+1)+(y[1:]**2)**(y[:-1]**2+1))

def shifted_rotated_rosenbrock(y,bias=700):
    y = np.array(y)
    return np.sum((np.square(100*np.square(y[1:]-np.square(y[:-1])))+np.square(y[:-1]-1)))+bias

def random(y):
    y = np.array(y)
    return np.random.randint(1)*np.random.random()*1000+1000
