import constants as c 
import numpy as np
from scipy import stats

def p_between_von_mises(a, b, kappa, x):
    # Calculate the CDF values for A and B at x
    cdf_a = stats.vonmises.cdf(2*np.pi*x, kappa, loc=2*np.pi*a)
    cdf_b = stats.vonmises.cdf(2*np.pi*x, kappa, loc=2*np.pi*b)
    
    # Calculate the probability of A < x < B
    p_between = np.abs(cdf_b - cdf_a)
    
    return p_between


 #
def action_dist(a,b):
    s=0
    for i in range(len(a)):
        s+=((a[i]-b[i])/(list(c.actuator_ranges.values())[i][1]-list(c.actuator_ranges.values())[i][0]))**2
    return np.sqrt(s)

def normalize(name, value):
    #normalize the value to be between 0 and 1
    return (value - c.sensor_ranges[name][0]) / (c.sensor_ranges[name][1] - c.sensor_ranges[name][0])