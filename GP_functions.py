import tdt
import os
import numpy as np
import pandas as pd
import GPy

from scipy import signal, integrate, stats
from scipy import interpolate
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from mpl_toolkits import mplot3d
from itertools import product
from multi_unit_functions import *
from multi_unit_plotting_functions import *

# Updated July 20th, 2021

#--------------------------------------------------------------------------------------------------------------
# Functions to prepare data to feed to the GP
#--------------------------------------------------------------------------------------------------------------

def get_1d_data(heatmap, target):
    """ This function takes the data from a given heatmap and returns the MSE and does
    
    Inputs
    ------
    heatmap : Pandas pivot table [which is also a dataframe]
        32 x 32 pivot table containing every combination of stimulation electrode/recording electrode and a metric
        This is obtained from get_delta_heatmap() in multi_unit_plotting_functions
    
    target : 1d array
        The target to compare each entry of the heatmap to. It needs to be the same dimension as the heatmap (32)
        
    Outputs
    -------
    X : Nx1 array
        Contains [int] for every row. (int is from 1 to 32)
        
    Y : Nx1 array
        MSE for a given electrode
    """
    X = np.arange(32)
    
    Y = np.empty_like(target)
    for i in range(len(heatmap.columns)):
        Y[i] = mean_squared_error(target, heatmap[heatmap.columns[i]].values)
    
    return X.reshape(-1,1), Y.reshape(-1, 1)

def get_2d_data(heatmap, target, MSE=False, normalize=False):
    """ This function takes the data from a given heatmap and returns the a distance metric to create a 
    Y parameter for the Gaussian Process. 
    
    It will also provide coordinates to the X parameters (stimulation electrodes) such that adjacent electrodes
    in the stimulation floating micro array (FMA) are also adjacent coordinates. 
    Since there are 32 electrodes in the FMA, X will be a 8x4 array. 
        
    Inputs
    ------
    heatmap : Pandas pivot table [which is also a dataframe]
        32 x 32 pivot table containing every combination of stimulation electrode/recording electrode and a metric
        This is obtained from get_delta_heatmap() in multi_unit_plotting_functions.py
        This should correspond to a given brain area (PMd, PMv, Area 5, M1)
    
    target : 1d array
        The target to compare each entry of the heatmap to. It needs to be the same dimension as the heatmap (32)
        
    MSE : bool
        Code to use squared sum of differences (SSD) to build the objective function (Y), otherwise 
        mean squared error (MSE) will be used
        1/True for SSD
        0/False for MSE
    
    normalize : bool
        Code use to toggle normalizing the data. 1 if yes, 0 if not
    
    
    Outputs
    -------
    X : 32 x 2 array
        Each row contains [x-coord, y-coord] of a given electrode to match this spatial arrangement
        X array                           FMA arrangement
        24 | 16 | 8  | 0                  28 | 20 | 10 | 2 
        25 | 17 | 9  | 1       |-----\    29 | 21 | 11 | 3
        26 | 18 | 10 | 2       |      \   30 | 22 | 12 | 4
        27 | 19 | 11 | 3       |      /   31 | 23 | 13 | 5
        28 | 20 | 12 | 4       |-----/    32 | 24 | 14 | 6
        29 | 21 | 13 | 5                  33 | 25 | 15 | 7
        30 | 22 | 14 | 6                  34 | 26 | 16 | 8
        31 | 23 | 15 | 7                  35 | 27 | 17 | 9
    NOTE: in the FMA arrangement the cells are shifted, which do not fully represent the actual arrangement due to the placements of the grounds. 
    
    Y : Nx1 array
        Objective function for a given stimulation electrode. 
        It can be either the squared sum of differences (SSD) or the mean squared error (MSE).
        It can also be normalized or not depending on the inputs to the function
        The values are multiplied by -1 to turn it into a maximization problem
        
    """
    
    # X
    X = np.zeros((2, 32))
#     X[0, :] = np.arange(32)
    for i in range(32):
        if i < 8:
            X[0][i] = 3
            X[1][i] = 7-i
        elif 8 <= i < 16:
            X[0][i] = 2
            X[1][i] = 7- i%8
        elif 16 <= i < 24:
            X[0][i] = 1
            X[1][i] = 7- i%8
        elif 24 <= i < 32:
            X[0][i] = 0
            X[1][i] = 7- i%8
    
    # Y
    
    def SSD(target, col):
        # Squared Sum of Differences
        return ((target - col)**2).sum()
    
    Y = np.empty_like(target)
    for i in range(len(heatmap.columns)):
        if MSE:
            Y[i] = -mean_squared_error(target, heatmap[heatmap.columns[i]].values)
            
        else:
            Y[i] = -SSD(target, heatmap[heatmap.columns[i]].values)
      
    # Normalize Y
    if normalize:
        for i in range(len(Y)):
            Y[i] = (Y[i]-Y.min())/(Y.max()-Y.min())
        
    return X.T, Y.reshape(-1, 1)

#--------------------------------------------------------------------------------------------------------------
# Functions to fit the data to the GP using GPy
#--------------------------------------------------------------------------------------------------------------



def fit_GP(kernel, X, Y, constraints=None, bounded=True):
    """ This function fits the GP and returns the mean and st.dev of the fit
    
    Inputs
    ------
    kernel : the kernel of the GP
        This is the kernel to fit from the GP library
        I intentionally left this out of the function so it can be changed more easily
        
    X : 32 x 2 array
        Each row contains [x-coord, y-coord] of a given electrode to match this spatial arrangement
        Output from get_2d_data() in GP_functions.py. See the function for more details
        X is equivalently the search space for the data
        
    Y : Nx1 array
        Sum of squared differences for a stimulation from a given stimulation electrode's response to the target
        Output from get_2d_data() in GP_functions.py. See the function for more details
    
    constraints : 1 integer or array of len 2
        Values to restrict the length scale by
        if len() = 1: restrict both directions by the same
        if len() = 2: restrict each direction separately
        if left as none --> do not restrict
        
    bounded : bool
        Basically to use bounded constrains or fixed constrains. 
        
    
    Outputs
    -------
    This no longer returns this
  #  mu : Nx1 array
  #      Contains the mean of the GP fit
  #  std : Nx1 array
  #      Contains the standard deviation of the GP fit. 
  #      
    m : GPy.rbf model fit
        
    
    """
    # Fit the model
    
    m = GPy.models.GPRegression(X, Y, kernel)
    
    # Constraints 
    if constraints is not None:
        if bounded:
            if isinstance(constraints, int) or isinstance(constraints, float):
    #             print('bounding both by same')
                m.rbf.lengthscale[[0]].constrain_bounded(0.05, constraints, warning=False)
                m.rbf.lengthscale[[1]].constrain_bounded(0.05, constraints, warning=False)

            elif len(constraints) == 2:
    #             print('2 different bounds')
                m.rbf.lengthscale[[0]].constrain_bounded(0.05, constraints[0], warning=False)
                m.rbf.lengthscale[[1]].constrain_bounded(0.05, constraints[1], warning=False)
            else:
                print("There is a mistake with the constraints input, aborting code")
                return 
        else:
            if isinstance(constraints, int) or isinstance(constraints, float):
                m.rbf.lengthscale[[0]] = constraints
                m.rbf.lengthscale[[0]].constrain_fixed(warning=False)
                m.rbf.lengthscale[[1]] = constraints
                m.rbf.lengthscale[[1]].constrain_fixed(warning=False)
                
            elif len(constraints) == 2:
                m.rbf.lengthscale[[0]] = constraints[0]
                m.rbf.lengthscale[[0]].constrain_fixed(warning=False)
                m.rbf.lengthscale[[1]] = constraints[1]
                m.rbf.lengthscale[[1]].constrain_fixed(warning=False)
                
            else:
                print('There is a mistake with the constraints input, aborting code')
                return

    # Optimize hyperparameters
    m.optimize()
#     m.optimize_restarts(num_restarts=10, verbose=False)
    
    return m



def predict_GP(model, search_space):
    """ Simple function to do the prediction for the GP model
    
    Inputs
    ------
    model : GP model
        This is the output from fit_GP()
        
    search_space: Nx2 array
        N <32. This is the search space that contains the coordinates of the search space
        
    Outputs
    -------
     mu : Nx1 array
        Contains the mean of the GP prediction
    std : Nx1 array
        Contains the standard deviation of the GP prediction. 
    """
    return model.predict(search_space)

#--------------------------------------------------------------------------------------------------------------
# Functions used to start an initial random search
#--------------------------------------------------------------------------------------------------------------

def build_random_search(X, Y, n_queries=1):
    """ Function to build an initial random search dataset
    
    Inputs
    ------
    X : 32 x 2 array
        Each row contains [x-coord, y-coord] of a given electrode to match this spatial arrangement
        Output from get_2d_data() in GP_functions.py. See the function for more details
        X is equivalently the search space for the data
        
    Y : Nx1 array
        Sum of squared differences for a stimulation from a given stimulation electrode's response to the target
        Output from get_2d_data() in GP_functions.py. See the function for more details
    
    n_queries : int
        Number of search points to randomly query
        
    Outputs
    -------
    x_search (n_queries x 2 data set):
        Contains the randomly queried points in the search space
    """
    if not isinstance(n_queries, int):
        print("'n_queries' needs to be an int. Aborting")
        return None
    
    # Create an initial search space
    x_search, y_search = [], []

    index = np.random.choice(X.shape[0], size=n_queries, replace=False)
    x_search.append(X[index])
    y_search.append(Y[index])
        
    return np.array(x_search).reshape(-1, X.shape[1]), np.array(y_search).reshape(-1, 1)


def append_random_search(x_search, y_search, X, Y, replace=True):
    """ Function to append a new point to the random search dataset
    
    If using this without replacement, there is a lot of list comprehension to get it to work. 
    A quick summary is that each row of X and x_search are made into strings and then any matching 
    strings are removed from X. Then X is converted back to an array with floats without the rows that
    are seen in x_search. 
    
    Inputs
    ------
    x_search : N x 2 array
        N is the number of randomly queried points (and any additional appended points)
        It is the input to the GP
    
    y_search : N x 1 array
        Y is the parameter that the GP is trying to fit on, each value of Y corresponds to one row of x_search
        
    X : 32 x 2 array
        Each row contains [x-coord, y-coord] of a given electrode to match this spatial arrangement
        Output from get_2d_data() in GP_functions.py. See the function for more details
        X is equivalently the search space for the data
        
    Y : Nx1 array
        Sum of squared differences for a stimulation from a given stimulation electrode's response to the target
        Output from get_2d_data() in GP_functions.py. See the function for more details
        
    replace : bool
        Append an X, Y paring that was not previously included in x_search, or y_search if true.
            
    Outputs
    -------
    Same as inputs with an appended search query
    """
    if replace:
        if x_search.shape[0] == X.shape[0]:
            print('The entire search space has been included, it is not possible to add another point')
            return 
        else:
            # Convert X and x_search into a matrix of strings
            string_X = np.array([str(i) for i in X])
            string_xsearch = np.array([str(i) for i in x_search])
            
            # This next line is easier to understand like this;
            # After for : for each item in string(X) not in string(x_search), convert back into a numpy float array
            # Inside np.fromstring, we have strings like '[1. 0.]'. item[1:-1] removes brackets. sep ' ' gives back 2d array
            X = np.array([np.fromstring(item[1:-1], sep=' ')  for item in string_X if item not in string_xsearch])
    
    index = np.random.choice(X.shape[0])
    x_search = np.append(x_search, X[index])
    y_search = np.append(y_search, Y[index])
    
    return x_search.reshape(-1, 2), y_search.reshape(-1, 1)

#--------------------------------------------------------------------------------------------------------------
# Functions for sequential GP (in 1D)
#--------------------------------------------------------------------------------------------------------------

# UCB function from Leo
def UCB(model, search_space, alpha=2):
    if len(search_space.shape)<2:
        mu, std = model.predict(search_space[:, None])
    else:
        mu, std = model.predict(search_space)
    return mu + alpha*std

 # nextX function from Leo
def nextX(model, search_space, alpha=2):
    return UCB(model, search_space, alpha).argmax()

def append_nextXY(X, Y, x_search, y_search, model, search_space, alpha=2):
    """
    Written to simplify code later. It will find and append the next point in a sequential GP fit. 
    
    Inputs
    ------
     X : 32 x 2 array
         Each row contains [x-coord, y-coord] of a given electrode to match this spatial arrangement
         Output from get_2d_data() in GP_functions.py. See the function for more details
         X is equivalently the search space for the data
        
     Y : Nx1 array
         Sum of squared differences for a stimulation from a given stimulation electrode's response to the target
         Output from get_2d_data() in GP_functions.py. See the function for more details
         i.e. the objective function of the GP
         
     x_search : N x 2 array
         N is the number of randomly queried points (and any additional appended points)
         It is the input to the GP
    
     y_search : N x 1 array
         Y is the parameter that the GP is trying to fit on, each value of Y corresponds to one row of x_search
         i.e. the objective function
         
     model : GPy.models.GPRegression object
         Contains the current fit of the model to determine the next search point in the acquisition function
         
     search_space : 32 x 2 array (in this scenario)
         Contains all possible values for the x_search. (Usually the same as X)
         
     alpha : float
         A hyperparameter for the UCB acquisition function. 
        
    
    Outputs
    -------
    x_search : (N+1) x 2 array
        Same as input but including the next point
    y_search : (N+1) x 1 array
        Same as input but including the objective function value for the next point
    """
    # Get the next x based on the UCB (see nextX and UCB functions)
    next_idx = nextX(model, search_space, alpha=alpha)
    
    # Append
    x_search = np.append(x_search, X[next_idx])
    y_search = np.append(y_search, Y[next_idx])
    
    return x_search.reshape(-1, 2), y_search.reshape(-1, 1)


#--------------------------------------------------------------------------------------------------------------
# Functions for sequential 'parallel' GPs
#--------------------------------------------------------------------------------------------------------------

def acquisition_2D(model_pmd, model_pmv, search_space, alpha=2E-3, epsilon=0.3):
    """
    This works by considering the UCB for 2 models and picking the next point out of both. 
    More contigencies may be useful later
    
    Inputs
    ------
    model_pmv : GPy.model.GPRegression object
        model fitted to the PMv data
        
    model_pmd : GPy.model.GPRegression object
        model fitted to the PMd data
        
    search_space : 32 x 2 array
        Contains all the possible values for the x_search. (Usually the same as X)
        Note that the search space for both models is the same but the objective function values (Y) are not the same
        
    alpha : float
        A hyperparameter for the UCB acquisition function
    
    epsilon : float [0, 1]
        Used for a greedy epsilon implementation to determine the next point. 
        
    Outputs
    -------
    location : string
        Contains a string saying "PMd" or "PMv" based on which area is selected for the next query
    
    next_ : 1 x 2 array
        Contains the coordinates for the next search in a given brain area. 
    
    """
    # Get the choices for the next point in both models by using a greedy epsilon-like implementation
    # Implementation inspired by James LeDoux's greedy epsilon's version
    use_eps_greedy = np.random.binomial(1, epsilon)
    
    if use_eps_greedy:
        next_choice = np.random.choice([0, 1])
    
    else:
    # Choose the maximum between both
        next_pmv = np.max(UCB(model_pmv, search_space, alpha))
        next_pmd = np.max(UCB(model_pmd, search_space, alpha))
        next_choice = np.argmax((next_pmv, next_pmd))

    if next_choice == 0:
        location = 'PMv'
        next_ = nextX(model_pmv, search_space, alpha)
    elif next_choice == 1:
        location = 'PMd'
        next_ = nextX(model_pmd, search_space, alpha)
    
    return location, next_

def append_next_query(search_space, Y_pmd, Y_pmv, x_search, y_search, locations, model_pmv, model_pmd, alpha=2E-3, epsilon=0.3):
    """ Written to simplify code later. It will find and append the next point in a sequential GP fit. 
        This calls aquisition_2D(). The details of the acquisition function are left in this aforementioned function
    
    Inputs
    ------
    search_space : 32 x 2 array
        Contains all the possible values for the x_search. (Usually the same as X)
        Note that the search space for both models is the same but the objective function values (Y) are not the same
    
    Y_pmd/Y_pmv : 32 x 1 arrays
        Contains the objective functions for the PMd and PMv respectively. 
        They should be obtained from get_2d_data(), which is defined in GP_functions.py
     
     x_search : N x 2 array
         N is the number of randomly queried points (and any additional appended points)
         It is the input to the GP
    
     y_search : N x 1 array
         Y is the parameter that the GP is trying to fit on, each value of Y corresponds to one row of x_search
         i.e. the objective function    
    
    locations : N x 1 array of strings
        Contains the location of stimulation for the GP. It is used to keep track of which stimulation has been used. 
        
    model_pmv, model_pmd : GPy.models.GPRegression objects
        Contains the models fitted to the PMv, PMd data respectively
        
    alpha : float
        A hyperparameter for the UCB acquisition function
        
    add_noise : bool
        Adding noise to the model to generate synthetic 
        
    Outputs
    -------
    x_search : (N+1) x 2 array
        Same as input but including the next point
    y_search : (N+1) x 1 array
        Same as input but including the objective function value for the next point
    locations : (N+1) x 1 array
        Same as input but inding the next stimulation location
    """
    
    # Get the next point to stimulate and the location from the acquistion function
    next_loc, next_idx = acquisition_2D(model_pmd, model_pmv, search_space, alpha=alpha, epsilon=epsilon)
    
    # Append data
    locations = np.append(locations, next_loc)
    x_search = np.append(x_search, search_space[next_idx])
    
    if next_loc == 'PMv':
        y_search = np.append(y_search, Y_pmv[next_idx])
    elif next_loc == 'PMd':
        y_search = np.append(y_search, Y_pmd[next_idx])
        
    return x_search.reshape(-1, 2), y_search.reshape(-1,1), locations


#--------------------------------------------------------------------------------------------------------------
# Functions for extracting results from the GP fit
#--------------------------------------------------------------------------------------------------------------

def get_best_electrode(X, x_search):
    """ This function gets the best electrode number based on the last value in the array of queried coordinates.
    In general, this should be the converged value of the array of queried coordinates because the loop will generally
    stop after the same coordinate is queried 3 times in a row for a given brain area. 
    
    Inputs
    ------
    X : 32 x 2 array
        Each row contains [x-coord, y-coord] of a given electrode to match this spatial arrangement
        Output from get_2d_data() in GP_functions.py. See the function for more details
        X is equivalently the search space for the data
        
    x_search : N x 2 array
         N is the number of queried points (including an initial random selection)
         
    Output
    ------
    electrode : integer
        The electrode number of the best fitting electrode to stimulate based on the queried search
    """

    # Convert the best value and the search space to strings for comparison (easier for multiple dimensions)
    # If there is only one point in the array
    if np.shape(x_search) == (2,):
        converged = str(x_search)
    # Else take the last one    
    else:
        converged = str(x_search[-1])
    
    string_X = np.array([str(i) for i in X])

    # Find the index of the converged value in string_X
    ind = np.where(string_X == converged)[0][0]

    all_electrodes = [ 2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,\
                      20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
    
    return all_electrodes[ind]

def get_best_fit_values(electrode, location, PMd_heatmap, PMv_heatmap):
    """ This function returns an array of the values from the best fitting electrodes. 
    It is useful for plotting and should help to reduce redundant code.
    
    Inputs
    ------
    electrode : integer
        The electrode number of the best fitting electrode to stimulate based on the queried search
        Output from get_best_electrode()
        
    locations : N x 1 string array
        Contains the location of the queried point for the different GPs. 
        N is the number of queried points.
        
    PMd_heatmap & PMv_heatmap : Pandas pivot table [which is also a dataframe]
        32 x 32 pivot table containing every combination of stimulation electrode/recording electrode and a metric
        This is obtained from get_delta_heatmap() in multi_unit_plotting_functions.py
        
    Output
    ------
    fit_values : 32 x 1 pandas Series
        Contains a column corresponding to the correct electrode from the correct heatmap according to the last
        queried location/electrode
    """
    
    if location == 'PMd':
        return PMd_heatmap[electrode]
    else:
        return PMv_heatmap[electrode]