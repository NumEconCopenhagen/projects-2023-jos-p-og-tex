import numpy as np
import matplotlib.pyplot as plt
import sympy as sm
from types import SimpleNamespace
from scipy import optimize



######## Question 2 ########

# Define profit function
def profit(kappa, par2):
    """ Find optimal labor to maximize profit """

    # a. solve
    obj = lambda l: -(kappa*l**(1-par2.eta)-par2.w*l)
    x0 = [0.0]
    res = optimize.minimize(obj,x0,method='L-BFGS-B')
        
    # b. save
    l_star = res.x[0]

    return l_star