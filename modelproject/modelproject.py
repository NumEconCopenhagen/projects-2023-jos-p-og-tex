from types import SimpleNamespace
import numpy as np
from scipy import optimize

import matplotlib.pyplot as plt
plt.rcParams.update({"axes.grid":True,"grid.color":"black","grid.alpha":"0.25","grid.linestyle":"--"})
plt.rcParams.update({'font.size': 14})



def solve(obj_lss,return_res=False):
    result = optimize.root_scalar(obj_lss,bracket=[0.1,1000],method='brentq')
    L_ss = result.root
    if return_res==True:
        return L_ss
    else:
        print(f'The steady state for L is: {result.root:.2f}')



def phase_diagram(par):

    # Number of observations 
    N = 100

    # Max value of L_t
    L_max = 10

    # Create a vector x1 from 0 to x_max, with N values
    L_vec = np.linspace(0,L_max,N)

    # Create an empty vector to store values of L_t+1
    L2_vec = np.empty(N)

    def L_func(lss,par=par):
        return ((1-par.beta)/par.lamb)*lss**(1-par.alpha)*(par.A*par.X)**(par.alpha)+(1-par.mu)*lss

    # Fill out out the vector
    for i, lss in enumerate(L_vec):
        L2_vec[i] = L_func(lss)

    # Create an empty vector to store values of L_t+1
    L3_vec = np.empty(N)

    # Fill out out the vector
    for i, lss in enumerate(L_vec):
        L3_vec[i] = lss

    # a. create the figure
    fig = plt.figure()

    # b. plot
    ax = fig.add_subplot(1,1,1)
    ax.plot(L_vec,L2_vec)
    ax.plot(L_vec,L3_vec)

    ax.set_title('The phase diagram for labor in the Malthus model')
    ax.set_xlabel('$L_{t}$')
    ax.set_ylabel('$L_{t+1}$')



def convergence(beta,lamb,mu,alpha,A_val,X_val,T_val,interactive=False):

    L_path = np.zeros(T_val)  # initialize a vector to store optimal L for each time period
    L_path[0] = 0.1  # set the initial value of L in the vector

    for i in range(1,T_val):

        # a. find next period L
        L_next = ((1-beta)/lamb)*L_path[i-1]**(1-alpha)*(A_val*X_val)**(alpha)+(1-mu)*L_path[i-1]

        # b. store value
        L_path[i] = L_next
    
    T_vec = np.linspace(0,T_val,T_val)
    fig = plt.figure()

    obj_lss = lambda lss: lss - (((1-beta)/lamb)*lss**(1-alpha)*(A_val*X_val)**(alpha)+(1-mu)*lss)

    if interactive==False:
        # c. plot
        ax = fig.add_subplot(1,1,1)
        ax.plot(T_vec,L_path)
        ax.axhline(solve(obj_lss,return_res=True),ls='--',color='black',label='analytical steady state')
        ax.set_title('The convergence diagram for the Malthus model')
        ax.set_xlabel('Time period')
        ax.set_ylabel('Labor force/population')

    else:
        # c. plot
        ax = fig.add_subplot(1,1,1)
        ax.plot(T_vec,L_path)
        ax.set_title('Convergence diagram for the Malthus model')
        ax.set_xlabel('Time period')
        ax.set_ylabel('Labor force/population')



def convergence_tech_shock(par,A_path):

    L_path = np.zeros(par.T)  # initialize a vector to store optimal L for each time period
    L_path[0] = 0.1  # set the initial value of L in the vector

    for i in range(1,par.T):

        # a. find next period L
        L_next = ((1-par.beta)/par.lamb)*L_path[i-1]**(1-par.alpha)*(A_path[i]*par.X)**(par.alpha)+(1-par.mu)*L_path[i-1]

        # b. store value
        L_path[i] = L_next
    
    T_vec = np.linspace(0,par.T,par.T)
    fig = plt.figure()

    obj_lss = lambda lss: lss - (((1-par.beta)/par.lamb)*lss**(1-par.alpha)*(par.A*par.X)**(par.alpha)+(1-par.mu)*lss)
    
    # c. plot
    ax = fig.add_subplot(1,1,1)
    ax.plot(T_vec,L_path)
    ax.axhline(solve(obj_lss,return_res=True),ls='--',color='black',label='analytical steady state without shock')
    ax.set_title('Convergence diagram for the Malthus model')
    ax.set_xlabel('Time period')
    ax.set_ylabel('Labor force/population')

    return L_path



def convergence_extension(par):

    L_path = np.zeros(par.T)  # initialize a vector to store optimal L for each time period
    L_path[0] = 0.1  # set the initial value of L in the vector
    A_path = np.zeros(par.T)
    A_path[0] = 1  # set the initial value of A in the vector
    A_path = np.power(1.02, np.arange(par.T))  # create growth in A

    for i in range(1,par.T):

        # a. find next period L
        L_next = ((1-par.beta)/par.lamb)*L_path[i-1]**(1-par.alpha)*(A_path[i]*par.X)**(par.alpha)+(1-par.mu)*L_path[i-1]

        # b. store value
        L_path[i] = L_next
    
    T_vec = np.linspace(0,par.T,par.T)
    fig = plt.figure()


    # c. plot
    ax = fig.add_subplot(1,1,1)
    ax.plot(T_vec,L_path)
    ax.set_title('Convergence diagram for the Malthus model')
    ax.set_xlabel('Time period')
    ax.set_ylabel('Labor force/population')
