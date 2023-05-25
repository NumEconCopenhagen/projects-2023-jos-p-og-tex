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

        # Calculate optimal labor supply
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
        delta0 = 0.11

        # Solve for optimal delta
        res = optimize.minimize(obj,delta0,method='Nelder-Mead',tol=1e-8)
            
        # Save result
        delta_star = res.x[0]

        # Print result
        if do_print:
            print(f'optimal delta: {delta_star:.3f}')
        else:
            return delta_star
        

        
    # Create plot for ex ante value given delta
    def delta_plot(self):
        
        # Generate delta values
        delta_values = np.linspace(0.001, 0.999, 100)

        # Create empty list to store the ex ante value
        value_vec = []

        # Calculate ex ante value for each delta value
        for delta_val in delta_values:
            value_val = self.ex_ante_value(10,self.l_vec2(delta_val))
            value_vec.append(value_val)

        # Find optimal delta
        opt_delta = self.value_opt()
        
        # Find ex ante value for optimal delta
        val_opt_delta = self.ex_ante_value(10,self.l_vec2(opt_delta))

        # Create plot
        fig = plt.figure(figsize=(7,4))
        ax = fig.add_subplot(1,1,1)
        ax.plot(delta_values, value_vec, label='Value')
        ax.scatter(opt_delta, val_opt_delta, label='Optimal delta')

        # Set labels and title
        ax.set_xlabel('Delta')
        ax.set_ylabel('Value')
        ax.set_title('Value of hair salon for different delta values')

        # Add a legend
        ax.legend()

        # Show the plot
        plt.show();

    # Suggest alternative policy
    def l_vec3(self,factor):

        par2 = self.par2

        # demand shock vector
        log_kappa = self.demand_shock()

        # Calculate optimal labor supply
        l_star = (((1-par2.eta)*np.exp(log_kappa))/par2.w)**(1/par2.eta)

        # Define new policy vector
        l_vec3 = np.zeros_like(l_star)
        l_vec3[0] = 0
        for i in range(1, len(l_vec3)):
            if i < 12:
                l_vec3[i] = l_star[i]*factor
            else:
                l_vec3[i] = l_star[i]
        return l_vec3


######## Question 3 ########

class question3:

    # Definition of parameters
    def __init__(self):
        """ setup model """

        # create namespaces
        sett = self.sett = SimpleNamespace()

        # settings
        sett.bounds = [-600, 600]
        sett.tolerance = 1e-8
        sett.warmup_iters = 10
        sett.max_iters = 1000


    def griewank_(self,x1,x2):
        A = x1**2/4000 + x2**2/4000
        B = np.cos(x1/np.sqrt(1))*np.cos(x2/np.sqrt(2))
        return A-B+1
    
    
    def griewank(self,x):
        return self.griewank_(x[0],x[1])
    

    def refined_global_optimizer(self,bounds, tolerance, warmup_iters, max_iters):
                
        # Step 1: Choose bounds for x and tolerance
        x_bounds = bounds
        tau = tolerance
        
        # Step 2: Choose warm-up and maximum iterations
        K_warmup = warmup_iters
        K_max = max_iters
        
        # Step 3: Iterations

        # Initialize x_star
        x_star = None

        # Create empty list to store x_k0
        x_k0_vec = []

        for k in range(K_max):
            # Step 3A: Draw random x^k uniformly within chosen bounds
            x_k = np.random.uniform(x_bounds[0], x_bounds[1], size=2)
            
            if k < K_warmup:
                # Step 3E: Run optimizer with x^k as initial guess
                res = optimize.minimize(self.griewank, x_k, method='BFGS', tol=tau)
                x_k_star = res.x
                x_k0_vec.append(x_k_star)

            else:
                # Step 3C: Calculate chi^k
                chi_k = 0.5 * (2 / (1 + np.exp((k - K_warmup) / 100)))
                
                # Step 3D: Calculate x_k0
                x_k0 = chi_k * x_k + (1 - chi_k) * x_star
                x_k0_vec.append(x_k0)
                
                # Step 3E: Run optimizer with x_k0 as initial guess
                res = optimize.minimize(self.griewank, x_k0, method='BFGS', tol=tau) 
                x_k_star = res.x
            
            # Step 3F: Update x_star if necessary
            if k==0 or self.griewank(x_k_star) < self.griewank(x_star):
                x_star = x_k_star
            
            # Step 3G: Check termination condition
            if self.griewank(x_star) < tau:
                nit = k
                break
        
        # Step 4: Return the result x_star
        return x_star, x_k0_vec, nit
        

    def plot_starting_guess(self):

        # Define bounds, tolerance, warmup and max iterations
        bounds = [-600, 600]
        tolerance = 1e-8
        warmup_iters = 10
        max_iters = 1000
        
        # Set seed and run optimizer
        np.random.seed(300)
        result = self.refined_global_optimizer(bounds, tolerance, warmup_iters, max_iters)

        # Extract x_k0_vec
        x_k0_vec = result[1]
        
        # Split in x1 and x2
        x1_vec = [point[0] for point in x_k0_vec]  # Extract the first element (x1) from each point
        x2_vec = [point[1] for point in x_k0_vec]  # Extract the second element (x2) from each point

        # Create plot
        fig = plt.figure(figsize=(7,4))
        ax = fig.add_subplot(1,1,1)
        plt.scatter(np.arange(len(x1_vec)), x1_vec, label='x1')
        plt.scatter(np.arange(len(x2_vec)), x2_vec, label='x2')

        # Set labels and title
        ax.set_xlabel('Iteration number')
        ax.set_ylabel('Value for x1 and x2')
        ax.set_title('Variation in effective intital guess for each iteration')

        # Add a legend
        ax.legend()

        # Show the plot
        plt.show();