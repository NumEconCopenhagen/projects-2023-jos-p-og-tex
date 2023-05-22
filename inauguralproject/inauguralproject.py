
# Import of packages 

from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd 
import matplotlib.pyplot as plt


# Modelspecification and class set-up e.g. HouseholdSpecializationModel.py
class HouseholdSpecializationModelClass:


# Definition of parameters
    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 

        # c. household production
        par.alpha = 0.5
        par.sigma = 1.0

        # d. wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

        # g. extension to the model - parameter
        par.k = 0



# Defining the utility function
    def calc_utility(self,LM,HM,LF,HF):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production
        if par.sigma == 1:
            H = HM**(1-par.alpha)*HF**par.alpha
        elif par.sigma == 0:
            H = min(HM,HF)
        else:
            H = ((1-par.alpha)*HM**((par.sigma-1)/par.sigma) + par.alpha*HF**((par.sigma-1)/par.sigma))**(par.sigma/(par.sigma-1)) 

        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutility of work
        epsilon_ = 1+1/par.epsilon
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_)
        
        return utility - disutility


# Defining the solution for discrete time
    def solve_discrete(self,do_print=False):
        """ solve model discretely """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # a. all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
    
        LM = LM.ravel() # vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt


# Defining the solution for continous time
    def solve(self,do_print=False):
        """ solve model continously """

        par = self.par 
        opt = SimpleNamespace()  

        # a. objective function (to minimize) - including penalty to account for time constraints (for Nelder-Mead method)
        def obj(x):
            LM,HM,LF,HF=x
            penalty=0
            time_M = LM+HM
            time_F = LF+HF
            if time_M > 24 or time_F > 24:
                penalty += 1000 * (max(time_M, time_F) - 24)
            return -self.calc_utility(LM,HM,LF,HF) + penalty
        
        # b. call solve
        x0=[2,2,2,2] # initial guess
        result = optimize.minimize(obj,x0,method='Nelder-Mead')
        
        # d. save results
        opt.LM = result.x[0]
        opt.HM = result.x[1]
        opt.LF = result.x[2]
        opt.HF = result.x[3]
        #opt.u = self.calc_utility(opt.LM,opt.HM,opt.LF,opt.HF)
        
        return opt


    def solve_wF_vec(self,discrete=False):
        """ solve model for vector of female wages """

        par = self.par
        sol = self.sol

        for i,wF in enumerate(par.wF_vec):    
            par.wF = wF
            
            # Running the model and replacing the values in the vectors with the optimal values
            opt = self.solve()
            sol.HF_vec[i]=opt.HF
            sol.HM_vec[i]=opt.HM

        return sol.HF_vec, sol.HM_vec


    def run_regression(self):
        """ run regression """

        par = self.par
        sol = self.sol

        self.solve_wF_vec()

        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec)
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0] 

        return sol.beta0,sol.beta1

    
    #### QUESTION 4
    def fit_data(self):
        """ find alpha and sigma to match data """

        par = self.par 
        sol = self.sol

        def objective(params):
            alpha, sigma = params
            par.alpha = alpha
            par.sigma = sigma

            # Assign the parameter values and define the target values for beta0 and beta1
            beta0 = 0.4
            beta1 = -0.1

            # Run the regression for the different vector of ratios between home production and wages when the parameters vary
            sol.beta0,sol.beta1 = self.run_regression()

            # Compute the objective value
            val = (beta0 - sol.beta0)**2 + (beta1 - sol.beta1)**2

            return val

        # Initial guess for alpha and sigma
        initial_guess = [0.5, 0.5]

        # Optimization
        result = optimize.minimize(objective, initial_guess, method='Nelder-Mead')

        # Extract the optimized values
        sol.optimal_alpha = result.x[0]
        sol.optimal_sigma = result.x[1]

        # Printing the results
        print(f"Minimum value: {result.fun:.2f}")
        print(f"Optimal alpha: {sol.optimal_alpha:.2f}")
        print(f"Optimal sigma: {sol.optimal_sigma:.2f}")


#### QUESTION 5
    # Defining a new utility function with preferences for who in the couple works at home
    def calc_utility5(self,LM,HM,LF,HF):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production
        if par.sigma == 1:
            H = HM**(1-par.alpha)*HF**par.alpha
        elif par.sigma == 0:
            H = min(HM,HF)
        else:
            H = ((1-par.alpha)*HM**((par.sigma-1)/par.sigma) + par.alpha*HF**((par.sigma-1)/par.sigma))**(par.sigma/(par.sigma-1)) 
        
        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutility of work
        epsilon_ = 1+1/par.epsilon
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_)

        # e. preferences for housework
        pref_H_work = par.k*np.log(HF/HM - HM/HF)
        
        return utility - disutility + pref_H_work
    
    # Solving the model as before, but know using the extended version
    def solve_extension(self):
        """ solve model continously """

        par = self.par 
        opt = SimpleNamespace()  

        # a. objective function (to minimize) - including penalty to account for time constraints (for Nelder-Mead method)
        def obj(x):
            LM,HM,LF,HF=x
            penalty=0
            time_M = LM+HM
            time_F = LF+HF
            if time_M > 24 or time_F > 24:
                penalty += 1000 * (max(time_M, time_F) - 24)
            return -self.calc_utility5(LM,HM,LF,HF) + penalty

        # b. call solve
        x0=[2,2,2,2] # initial guess
        result = optimize.minimize(obj,x0,method='Nelder-Mead')
        
        # d. save results
        opt.LM = result.x[0]
        opt.HM = result.x[1]
        opt.LF = result.x[2]
        opt.HF = result.x[3]
        
        return opt

    # Solving the model for a vector of female wages
    def solve_wF_vec5(self,discrete=False):
        """ solve model for vector of female wages """

        par = self.par
        sol = self.sol
    
        for i,wF in enumerate(par.wF_vec):
            par.wF = wF
            
            # Running the model and replacing the values in the vectors with the optimal values
            opt = self.solve_extension()
            sol.HF_vec[i]=opt.HF
            sol.HM_vec[i]=opt.HM

        return sol.HF_vec, sol.HM_vec
    

    def run_regression5(self):
        """ run regression """

        par = self.par
        sol = self.sol

        self.solve_wF_vec5()

        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec)
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0] 

        return sol.beta0,sol.beta1


    # Defining a method to find the optimal value of k to match the target values for beta0 and beta1
    def optimal_k(self):
        """ find alpha and sigma to match data """

        par = self.par 
        sol = self.sol

        def objective_new(k):
            par.k = k

            # Assign the parameter values and define the target values for beta0 and beta1
            beta0 = 0.4
            beta1 = -0.1

            # Run the regression for the different vector of ratios between home production and wages when the parameters vary
            sol.beta0,sol.beta1 = self.run_regression5()

            # Compute the objective value
            val = (beta0 - sol.beta0)**2 + (beta1 - sol.beta1)**2

            return val

        # Initial guess for k
        initial_guess_k = 0.0005

        # Optimization
        result = optimize.minimize(objective_new, initial_guess_k, method='Nelder-Mead')

        # Extract the optimized values
        sol.optimal_k = result.x[0]

        # Printing the results
        print(f"Minimum value: {result.fun:.2f}")
        print(f"Optimal k: {sol.optimal_k:.5f}")
        print(f"Beta0: {sol.beta0:.2f}")
        print(f"Beta1: {sol.beta1:.2f}")