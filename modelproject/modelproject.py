from types import SimpleNamespace
import numpy as np
from scipy import optimize

class MalthusModelClass():

    def __init__(self,do_print=True):
        """ create the model """

        if do_print: print('initializing the model:')

        self.par = SimpleNamespace()
        self.ss = SimpleNamespace()
        self.path = SimpleNamespace()

        if do_print: print('calling .setup()')
        self.setup()

        if do_print: print('calling .allocate()')
        self.allocate()
    

    def setup(self):
        """ baseline parameters """

        par = self.par

        # a. household
        par.beta = np.nan
        par.lambd = np.nan

        # b. firms
        par.alpha = 0.30    
        par.mu = 0.2
        par.A = 1
        par.X = 1

        # c. initial
        par.L_lag_ini = 1.0

        # d. misc
        par.Tpath = 500 # length of transition path, "truncation horizon"


    def allocate(self):
        """ allocate arrays for transition path """
        
        par = self.par
        path = self.path

        allvarnames = ['Lt1','beta','lambd','Lt','alpha','A','X','mu']
        for varname in allvarnames:
            path.__dict__[varname] =  np.nan*np.ones(par.Tpath)


    def find_steady_state(self,L_ss,do_print=True):
        """ find steady state """

        par = self.par
        ss = self.ss

        # a. find L
        ss.L = L_ss
        Y,_,_ = production(par,ss.L)

        if do_print:

            print(f'L_ss = {ss.L:.4f}')
            print(f'beta = {par.beta:.4f}')


    def evaluate_path_errors(self):
        """ evaluate errors along transition path """

        par = self.par
        ss = self.ss
        path = self.path
        
        # a. labor
        L = path.L
        L_lag = path.L_lag = np.insert(L[:-1],0,par.L_lag_ini)

        # b. errors (also called H)
        errors = np.nan*np.ones((par.Tpath))
        errors = L - ((1-par.beta)/par.lambd)*L_lag**(1-par.alpha)*(A*X)**par.alpha + (1-par.mu)*L_lag
        
        return errors
        
        
    def solve(self,do_print=True):
        """ solve for the transition path """

        par = self.par
        ss = self.ss
        path = self.path
        
        # a. equation system
        def eq_sys(x):
            
            # i. update
            x = x.reshape(par.Tpath)
            path.L = x
            
            # ii. return errors
            return self.evaluate_path_errors()

        # b. initial guess
        x0 = np.nan*np.ones(par.Tpath)
        x0 = ss.L

        # c. call solver    
        root = optimize.root_scalar(eq_sys,bracket=[0.1,1000],method='brentq')

        x = root.x
        
        # d. final evaluation
        eq_sys(x)

            
def production(par,L_lag):
    """ production """

    Y = L_lag**(1-par.alpha)*(par.A*par.X)**par.alpha
    return Y           