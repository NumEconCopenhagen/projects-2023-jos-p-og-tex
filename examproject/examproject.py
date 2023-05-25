import numpy as np
import matplotlib.pyplot as plt
import sympy as sm
from types import SimpleNamespace
from scipy import optimize



######## Question 2 ########

class question2:

# Definition of parameters
    def __init__(self):
        """ setup model """

        # create namespaces
        par2 = self.par2 = SimpleNamespace()

        # parameters for static model
        par2.eta = 0.5
        par2.w = 1

        # parameters for the dynamic model
        par2.rho = 0.90
        par2.iota = 0.01
        par2.sigma = 0.1
        par2.R = (1+0.01)**(1/12)

        # time periods
        par2.n = 120

 

    # Define profit function
    def profit(self,kappa):
        """ Find optimal labor to maximize profit """

        par2 = self.par2

        # a. solve
        obj = lambda l: -(kappa*l**(1-par2.eta)-par2.w*l)
        x0 = [0.0]
        res = optimize.minimize(obj,x0,method='L-BFGS-B')
            
        # b. save
        l_star = res.x[0]

        return l_star


    # Check if given solution matches numerical solution
    def check_numerical(self,kappa_vec):
        """ Check if analytical solution is optimal"""
        
        par2 = self.par2

        for kappa in kappa_vec:
            l_optimal = self.profit(kappa)
            l_given = ((1-par2.eta)*kappa/par2.w)**(1/par2.eta)
            print(f'kappa = {kappa:.1f} gives optimal l = {l_optimal:.2f} and given solution is {l_given:.2f}')


    # Define the demand-shock
    def demand_shock(self):
        """ Define path for demand shock"""
        
        par2 = self.par2

        # Define the error term

        # Mean and standard deviation of the distribution
        par2.mu = -0.5*par2.sigma**2 
        par2.sigma = par2.sigma  
        
        # Set a random seed for reproducibility
        np.random.seed(2805)

        # Generate the random shocks
        epsilon = np.random.normal(par2.mu, par2.sigma, par2.n)

        # Create an empty vector to store values of the demand shock
        log_kappa = []

        # Fill out out the vector given the shock path
        for i,eps in enumerate(epsilon):
            if i == 0:
                log_kappa.append(np.log(1))
            else:
                log_kappa.append(par2.rho*log_kappa[i-1]+eps)

        return log_kappa



    # Define ex post value of the hair salon
    def ex_post_value(self,log_kappa,l_path):
        """ Calculate ex post value of the hair salon"""

        par2 = self.par2

        # time vector
        t = np.linspace(0, par2.n, par2.n)

        result = 0
        for i in range(len(l_path)):
            if i > 0 and l_path[i] != l_path[i-1]:
                result += (par2.R**(-t[i])*(np.exp(log_kappa[i])*l_path[i]**(1-par2.eta)-par2.w*l_path[i]-par2.iota))
            else:
                result += (par2.R**(-t[i])*(np.exp(log_kappa[i])*l_path[i]**(1-par2.eta)-par2.w*l_path[i]))
        
        return result
    

    # Define ex ante value of the hair salon
    def ex_ante_value(self,K,l_path,do_print=False):
        """ Calculate ex ante value of the hair salon"""

        par2 = self.par2

        # create empty list to store values of the ex ante value
        H_sum = []

        # demand shock vector
        log_kappa = self.demand_shock()
        
        for k in range(0,K-1):
            np.random.seed(k)
            epsilon = np.random.normal(par2.mu, par2.sigma, par2.n)
            log_kappa_k = []
            for i,eps in enumerate(epsilon):
                if i == 0:
                    log_kappa_k.append(np.log(1))
                else:
                    log_kappa_k.append(par2.rho*log_kappa[i-1]+eps)
            h_k = self.ex_post_value(log_kappa_k,l_path)
            H_sum.append(h_k)

        H = 1/K*np.sum(H_sum)
        
        if do_print == True:
            print(f'The ex ante value of the hair salon is {H:.2f}')
        else:
            return H
        

    # Define new policy vector dependent on delta
    def l_vec2(self,delta):
        """ Calculate new policy vector for labor supply"""

        par2 = self.par2

        # demand shock vector
        log_kappa = self.demand_shock()

        l_star = (((1-par2.eta)*np.exp(log_kappa))/par2.w)**(1/par2.eta)

        l_vec2 = np.zeros_like(l_star)
        l_vec2[0] = 0
        for i in range(1, len(l_vec2)):
            if np.abs(l_vec2[i-1]-l_star[i]) > delta:
                l_vec2[i] = l_star[i]
            else:
                l_vec2[i] = l_vec2[i-1]
        return l_vec2


    # Define function
    def value_opt(self,do_print=False):
        """ Find optimal delta to maximize value function"""

        # Define objective function
        def obj(delta):
            return -self.ex_ante_value(10,self.l_vec2(delta))

        # Make initial guess for delta
        delta0 = [0.05]

        # Solve for optimal delta
        res = optimize.minimize(obj,delta0,method='Nelder-Mead')
            
        # Save result
        delta_star = res.x[0]

        # Print result
        if do_print:
            print(f'optimal delta: {delta_star:.3f}')
        else:
            return delta_star