import numpy as np
import matplotlib.pyplot as plt
import sympy as sm
from types import SimpleNamespace
from scipy import optimize

######## Question 1 ########

class question1:
    
    # Definition of parameters
    def __init__(self):
        """ setup model """

        # Create namespaces
        par = self.par = SimpleNamespace()

        # Parameters 
        par.kappa = 1.0
        par.alpha = 0.5
        par.v = 1/(2*16**2)
        par.w = 1.0
        par.tau = 0.3

        # Parameters for general model
        par.sigma1 = 1.001
        par.rho1 = 1.001
        par.epsilon1 = 1.0

        par.sigma2 = 1.5
        par.rho2 = 1.5
        par.epsilon2 = 1.0

    
    # Define method to plot how L depends on wage
    def plot_L_wage(self,L_star_func):
        
        par = self.par
        
        # Define a vector for values of real wage
        w_values = np.linspace(1, 20, 100)

        # Define a vector to store optimal labor
        L_vec = []

        # Calculate optimal labor for different values of the real wage
        for ws in w_values:
            tau = 0.3
            w_tilde_val = (1-tau)*ws
            L_val = L_star_func(par.kappa, par.alpha, par.v, w_tilde_val)
            L_vec.append(L_val)

        # Plot the figure
        fig1 = plt.figure(figsize=(9, 4))
        ax = fig1.add_subplot(1, 1, 1)

        ax.plot(w_values, L_vec, ls='-', lw=2, color='blue')

        ax.set_xlabel('wage (w)')
        ax.set_ylabel('Optimal labor supply ($L^*$)')
        ax.set_title('Relationship between labour supply and real wage \n');


    # Define method to plot implied L, G and worker utility for different tau values
    def plot_functions(self,L_star_func):

        par = self.par

        # Generate data for the plot
        tau_values = np.linspace(0.001, 0.999, 100)

        # Create empty lists to store the data
        L_vec = []
        G_vec = []
        tau_vec = []
        utility_vec = []

        for tau_val in tau_values:
            # Create vector of tau values
            tau_vec.append(tau_val)

            # Calculate w_tilde
            w_tilde_val = (1-tau_val)*par.w

            # Calculate optimal labor and store
            L_val = L_star_func(par.kappa, par.alpha, par.v, w_tilde_val)
            L_vec.append(L_val)

            # Calculate optimal government consumption and store
            G_val = tau_val*par.w*L_val*((1-tau_val)*par.w)
            G_vec.append(G_val)

            # Calculate worker utility and store
            u_val = np.log(G_val**(1-par.alpha)*(par.kappa+(1-tau_val)*par.w*L_val)) - par.v*(L_val**2/2)
            utility_vec.append(u_val)


        # Create plot
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(1,1,1)
        ax.plot(tau_vec, L_vec, label='Labor Supply')
        ax.plot(tau_vec, G_vec, label='Government Consumption')
        ax.plot(tau_vec, utility_vec, label='Utility')


        # Set labels and title
        ax.set_xlabel('Tax Rate (tau)')
        ax.set_ylabel('Government Consumption, Labor Supply, and Worker Utility')
        ax.set_title('Relationship between Tax Rate, Government Consumption, Labor Supply, and Worker Utility')

        # Add a legend
        ax.legend()

        # Show the plot
        plt.show();


    def utility(self,L_star_func,tau):
        """ Calculate utility """

        par = self.par
        par.tau = tau

        # Calculate optimal labor 
        L_val = L_star_func(par.kappa, par.alpha, par.v, par.w)

        # Calculate optimal government consumption 
        G_val = par.tau*par.w*L_val*((1-par.tau)*par.w)

        # Calculate worker utility and store
        u_val = np.log(G_val**(1-par.alpha)*(par.kappa+(1-par.tau)*par.w*L_val)) - par.v*(L_val**2/2)

        return u_val


    def max_utility(self,L_star_func,do_print=False):
        """ Find optimal tax rate to maximize utility function """

        par = self.par

        # Define objective function
        def obj(tau):
            par.tau = tau
            util = self.utility(L_star_func,tau)
            return -util

        # Make initial guess for delta
        tau0 = 0.2

        # Solve for optimal delta
        res = optimize.minimize(obj,tau0,method='Nelder-Mead',tol=1e-8)
            
        # Save result
        tau_star = res.x[0]

        # Print result
        if do_print:
            print(f'optimal tax rate: {tau_star:.3f}')
            
            # Create plot
            tau_values = np.linspace(0.001, 0.999, 100)
            tau_vec = []
            utility_vec = []

            for tau_val in tau_values:
                # Create vector of tau values
                tau_vec.append(tau_val)

                # Calculate worker utility and store
                u_val = self.utility(L_star_func,tau_val)
                utility_vec.append(u_val)


            fig = plt.figure(figsize=(10,8))
            ax = fig.add_subplot(1,1,1)
            ax.plot(tau_vec, utility_vec, label='Utility')
            ax.scatter(tau_star,self.utility(L_star_func,tau_star), label='Optimal tax rate', s=100) 


            # Set labels and title
            ax.set_xlabel('Tax Rate (tau)')
            ax.set_ylabel('Government Consumption, Labor Supply, and Worker Utility')
            ax.set_title('Relationship between Tax Rate and Worker Utility')

            # Add a legend
            ax.legend()

            # Show the plot
            plt.show();

        else:
            return tau_star
        

    def general_utility(self,G,L,tau,case=1):
        """ General utility function """

        par = self.par

        # Define C
        C = par.kappa+(1-tau)*par.w*L

        if case == 1:
            sigma = par.sigma1 
            rho = par.rho1
            epsilon = par.epsilon1
            return (((par.alpha*C**((sigma-1)/sigma)+(1-par.alpha)*G**((sigma-1)/sigma))**(sigma/(sigma-1)))**(1-rho)-1)/(1-rho) - par.v*(L**(1+epsilon)/(1+epsilon))
        else:
            sigma = par.sigma2
            rho = par.rho2
            epsilon = par.epsilon2
            return (((par.alpha*C**((sigma-1)/sigma)+(1-par.alpha)*G**((sigma-1)/sigma))**(sigma/(sigma-1)))**(1-rho)-1)/(1-rho) - par.v*(L**(1+epsilon)/(1+epsilon))


    def max_general_utility(self,L_star_func,G,case=1,do_print=False):
        """ Find optimal tax rate to maximize utility function """

        par = self.par

        # Define objective function
        if case == 1:
            def obj(L):
                tau_star = self.max_utility(L_star_func)
                util = self.general_utility(G,L,tau_star)
                return -util
        else:
            def obj(L):
                tau_star = self.max_utility(L_star_func)
                util = self.general_utility(G,L,tau=tau_star,case=2)
                return -util

        # Make initial guess for labour
        L0 = 0.2

        # Solve for optimal labour
        res = optimize.minimize(obj,L0,method='Nelder-Mead',tol=1e-8)
            
        # Save result
        L_star = res.x[0]
 
        return L_star
    

    # Define method to find optimal G given the tax rate
    def optimal_G(self,L_star_func,case=1,do_print=True):
        """ Find optimal tax rate to maximize utility function """

        par = self.par

        # Define objective function
        if case == 1:
            def obj(G):
                tau_star = self.max_utility(L_star_func)
                L_star1 = self.max_general_utility(L_star_func,G,case=1)
                return G-tau_star*par.w*L_star1
        else:
            def obj(G):
                tau_star = self.max_utility(L_star_func)
                L_star2 = self.max_general_utility(L_star_func,G,case=2)
                return G-tau_star*par.w*L_star2

        # Solve for optimal delta
        result = optimize.root_scalar(obj,bracket=[0.01, 7],method='brentq')
            
        # Save result
        G_star = result.root

        if do_print:
            print(f'Optimal G: {G_star:.2f}')
        else: 
            return G_star


    # Define method to find optimal tax rate  
    def optimal_tax(self,case=1):
        """ Find optimal tax rate to maximize utility function """
        
        par = self.par

        # Generate G value
        G_values = np.linspace(0.001, 7, 100)

        # Create empty list to store the ex ante value
        prev_utility = -np.inf
        opt_tax = None
        opt_labor = None
        Optimal_G = None

        # Utility function for the worker
        if case == 1:
            def obj(x):
                G_val,L_val,tau_val = x
                util = self.general_utility(G_val,L_val,tau_val)
                return -util
        else:
            def obj(x):
                G_val,L_val,tau_val = x
                util = self.general_utility(G_val,L_val,tau_val,case=2)
                return -util

        # Define constraint    
        def constraint_func(x):
            G_val,L_val,tau_val = x
            return G_val - tau_val*par.w*L_val
        
        constraint = {'type':'eq','fun':constraint_func}

        # Initial guess for the tax rate
        initial_guess = [5.0,10,0.4]

        # Bounds
        bound = [(0.01,20),(0.01,50),(0,1)]

        # Perform the optimization
        result = optimize.minimize(obj,initial_guess,method='SLSQP',constraints=constraint,bounds=bound)

        # Retrieve the optimal tax rate and labor
        optimal_tax_rate = result.x[2]
        optimal_labor_supply = result.x[1]
        optimal_gov_cons = result.x[0]

        # Evaluate the worker's utility at the optimal tax rate
        optimal_utility = -result.fun

        if optimal_utility > prev_utility:
            prev_utility = optimal_utility
            opt_tax = optimal_tax_rate
            opt_labor = optimal_labor_supply
            Optimal_G = optimal_gov_cons

        # Print the results
        print(f"Optimal Tax Rate: {opt_tax:.2f}")
        print(f"Maximum utility: {prev_utility:.2f}")
        print(f"Optimal Labor Supply: {opt_labor:.2f}")
        print(f"Optimal Government Consumption: {Optimal_G:.2f}")

######## Question 2 ########

class question2:

    # Definition of parameters
    def __init__(self):
        """ setup model """

        # Create namespaces
        par2 = self.par2 = SimpleNamespace()

        # Parameters for static model
        par2.eta = 0.5
        par2.w = 1

        # Parameters for the dynamic model
        par2.rho = 0.90
        par2.iota = 0.01
        par2.sigma = 0.1
        par2.R = (1+0.01)**(1/12)

        # Time periods
        par2.n = 120

 

    # Define profit function
    def profit(self,kappa):
        """ Find optimal labor to maximize profit """

        par2 = self.par2

        # Optimize
        obj = lambda l: -(kappa*l**(1-par2.eta)-par2.w*l)
        x0 = [0.0]
        res = optimize.minimize(obj,x0,method='L-BFGS-B')
            
        # Store results
        l_star = res.x[0]

        return l_star


    # Check if given solution matches numerical solution
    def check_numerical(self,kappa_vec):
        """ Check if analytical solution is optimal """
        
        par2 = self.par2

        # Calculate profit for different values of kappa
        for kappa in kappa_vec:
            l_optimal = self.profit(kappa)
            l_given = ((1-par2.eta)*kappa/par2.w)**(1/par2.eta)
            print(f'kappa = {kappa:.1f} gives optimal l = {l_optimal:.2f} and given solution is {l_given:.2f}')


    # Define the demand-shock
    def demand_shock(self):
        """ Define path for demand shock """
        
        par2 = self.par2

        ### Define the error term

        # Mean and standard deviation of the distribution
        par2.mu = -0.5*par2.sigma**2 
        par2.sigma = par2.sigma  
        
        # Set a random seed for reproducibility
        np.random.seed(2805)

        # Generate the random error terms
        epsilon = np.random.normal(par2.mu, par2.sigma, par2.n)

        ### Create path for demand shock

        # Create an empty vector to store values of the demand shock
        log_kappa = []

        # Fill out out the vector given the error term path
        for i,eps in enumerate(epsilon):
            if i == 0:
                log_kappa.append(np.log(1))
            else:
                log_kappa.append(par2.rho*log_kappa[i-1]+eps)

        return log_kappa



    # Define ex post value of the hair salon
    def ex_post_value(self,log_kappa,l_path):
        """ Calculate ex post value of the hair salo """

        par2 = self.par2

        # Set up time vector
        t = np.linspace(0, par2.n, par2.n)

        # Initialize result
        result = 0

        # Calulate ex post value for given shock path
        for i in range(len(l_path)):
            if i > 0 and l_path[i] != l_path[i-1]:
                result += (par2.R**(-t[i])*(np.exp(log_kappa[i])*l_path[i]**(1-par2.eta)-par2.w*l_path[i]-par2.iota))
            else:
                result += (par2.R**(-t[i])*(np.exp(log_kappa[i])*l_path[i]**(1-par2.eta)-par2.w*l_path[i]))
        
        return result
    

    # Define ex ante value of the hair salon
    def ex_ante_value(self,K,l_path,do_print=False):
        """ Calculate ex ante value of the hair salon """

        par2 = self.par2

        # Create empty list to store values of the ex ante value
        H_sum = []

        # Demand shock vector
        log_kappa = self.demand_shock()
        
        # Calculate values for each time period
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

        # Sum to ex ante value
        H = 1/K*np.sum(H_sum)
        
        # Print results
        if do_print == True:
            print(f'The ex ante value of the hair salon is {H:.2f}')
        else:
            return H
        

    # Define new policy vector dependent on delta
    def l_vec2(self,delta):
        """ Calculate new policy vector for labor supply """

        par2 = self.par2

        # Demand shock vector
        log_kappa = self.demand_shock()

        # Calculate optimal labor supply
        l_star = (((1-par2.eta)*np.exp(log_kappa))/par2.w)**(1/par2.eta)

        # Setup vector for new policy
        l_vec2 = np.zeros_like(l_star)
        
        # Set labor to 0 in initial period
        l_vec2[0] = 0

        # Update labor with new policy
        for i in range(1, len(l_vec2)):
            if np.abs(l_vec2[i-1]-l_star[i]) > delta:
                l_vec2[i] = l_star[i]
            else:
                l_vec2[i] = l_vec2[i-1]
        return l_vec2


    # Define function to find delta that maximize value
    def value_opt(self,do_print=False):
        """ Find optimal delta to maximize value function """

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
        """ Create plot for value given delta """
        
        # Generate delta values
        delta_values = np.linspace(0.001, 1, 100)

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
        """ Define new policy vector """

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

        # Create namespaces
        sett = self.sett = SimpleNamespace()

        # Settings
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
    

    # Define efined global optimizer with multi-start
    def refined_global_optimizer(self,bounds, tolerance, warmup_iters, max_iters):
        """ Global optimizer with multi-start for griewank function """   

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
                # Step 3E: Run optimizer with x_k as initial guess
                res = optimize.minimize(self.griewank, x_k, method='BFGS', tol=tau)
                x_k_star = res.x
                x_k0_vec.append(x_k_star)

            else:
                # Step 3C: Calculate chi_k
                chi_k = 0.5*(2/(1+np.exp((k-K_warmup)/100)))
                
                # Step 3D: Calculate x_k0
                x_k0 = chi_k*x_k+(1-chi_k)*x_star
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
        
        # Step 4: Return the result x_star, the vector for initial guesses and number of iterations
        return x_star, x_k0_vec, nit
        

    # Define method to plot initial guess given iteration
    def plot_starting_guess(self):
        """ Plot initial guess for each iteration """

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