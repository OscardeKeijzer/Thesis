# -*- coding: utf-8 -*-

# References:
#     [1] Brigo, D., & Mercurio, F. (2006). Interest Rate Models - Theory and 
#         Practice: With Smile, Inflation, and Credit. Springer, Berlin, 
#         Heidelberg. doi: 10.1007/978-3-540-34604-3
#     [2] Glasserman, P. (2003). Monte Carlo Methods in Financial Engineering. 
#         Springer. doi: 10.1007/978-0-387-21617-1

# Imports
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy.stats as st
from tqdm import trange

# Local imports
from data_functions import write_Parquet_data
from plotting_functions import (plot_one_factor_Hull_White_histogram, 
                                plot_time_series)

def gen_one_factor_Hull_White_paths(a_param: float,
                                    antithetic: bool,
                                    experiment_dir: str,
                                    n_annual_trading_days: int,
                                    n_paths: int,
                                    r_t_process_type: str,
                                    seed: int,
                                    sigma: float,
                                    sim_time: float,
                                    sim_type: str,
                                    time_0_dfdt_curve: list,
                                    time_0_f_curve: list,
                                    time_0_rate: list,
                                    verbose: bool = False
                                    ) -> list:
    """
    Info:
        This function generates short rate time series in the one-factor 
        Hull & White (1F-HW) model under the risk-neutral measure Q as 
        approximated using Euler discretization of the solution to the 1F-HW 
        stochastic differential equation.
        
    Input:
        a_param: the constant mean reversion rate parameter in the 1F-HW short 
            rate model.
        
        antithetic: a bool which if True, sets the zero-mean process  to be 
            simulated using antithetic standard normal random draws. 
            Note: this was only implemented for the shifted zero-mean process 
            simulation using the Euler method.
        
        experiment_dir: the directory to which the results are saved.
        
        n_annual_trading_days: an int specifying the number of trading days 
            per year for the time axis discretization.
            
        n_paths: an int specifying the number of simulated short rate paths.
        
        r_t_process_type: a str specifying whether the short rates will be 
            simulated directly ('direct') or as a shifted zero-mean process 
            ('zero-mean').
            
        seed: an int specifying the seed for the randon number generator used 
            for the generation of the short rate paths.
        
        sigma: the constant volatility factor in the 1F-HW short rate model.
        
        sim_time: the simulation time in years.
        
        sim_type: a str specifying the simulation discretization type: if 
            'Euler', the short rates are simulated using Euler-Maruyama 
            discretization; if 'exact', the short rates are simulated by 
            sampling the exact distribution.
            
        time_0_dfdt_curve: an ndarray containing the time derivative values of the time-zero 
            instantaneous forward curve.
            
        time_0_f_curve: an ndarray containing the time values of the market 
            time-zero instantaneous forward curve.
        
        time_0_rate: the initial interest rate on which the term structure is 
            based.
            
        verbose: a bool which if False blocks the function prints.
    
    Output:
        r_t_paths: a 2D ndarray containing the simulated short rate paths along 
        a number of columns corresponding to the number of paths and a number 
        of rows being a discrete short rate time series of length equal to the 
        total number of trading days.
        
        x_t_paths: if the short rates were simulated as a shifted zero-mean 
            process: a 2D ndarray containing the zero-mean process paths along 
            a number of columns corresponding to the number of paths and a 
            number of rows being a discrete short rate time series of length 
            equal to the total number of trading days.
    """
    if verbose:
        print(25*'*' + '\n1F-HW short rate function under Q with' 
              + (f' {sim_type}.capitalize()' 
                if r_t_process_type.lower() == 'Euler' 
                else f' {sim_type}') + ' discretization' 
              + ( ' using zero-mean process' 
                  if r_t_process_type.lower() == 'zero-mean' 
                  else '') + ' initialized.')
    
    ## Check whether short rate process type and simulation type were specified
    ## correctly
    if (r_t_process_type.lower() != 'direct' 
        and r_t_process_type.lower() != 'zero-mean'):
        raise ValueError('short rate process type not recognized. Enter as' 
                          + ' "direct" or "zero-mean".')
    if sim_type.lower() != 'euler' and sim_type.lower() != 'exact':
        raise ValueError('short rate simulation type not recognized. Enter as' 
                          + ' "Euler" or "exact".')
    
    ## Simulate short rate process
    # Determine the total number of trading days, time axis spacing delta_t, 
    # and the random number seed
    n_trading_days = int(sim_time*n_annual_trading_days)
    delta_t = sim_time/n_trading_days
    np.random.seed(seed) if seed is not None else np.random.seed()
    
    # Initialize array for storing short rate paths
    r_t_paths = np.zeros((n_trading_days+1, n_paths))
    r_t_paths[0] = time_0_rate
    
    if verbose:
        print(f'\nGenerating {n_paths:,} 1F-HW short rate paths...')
        
    if r_t_process_type.lower() == 'direct':
        if sim_type.lower() == 'euler':
            # Loop over all short rate paths
            for path in (trange(n_paths) if verbose else range(n_paths)):
                short_rate = np.zeros(n_trading_days+1)
                short_rate[0] = time_0_rate
                
                # Loop over all time points of the current path
                for t_idx in range(n_trading_days):
                    # Evaluate the time value in years of the current time 
                    # point
                    time_t = t_idx*delta_t
                    
                    # Evaluate the time-dependent drift parameter theta
                    theta_t = (time_0_dfdt_curve[t_idx] 
                               + a_param*time_0_f_curve[t_idx+1] 
                                + sigma**2/(2*a_param)
                                *(1 - np.exp(-2*a_param*time_t)))
                    
                    # Sample standard normal distribution
                    Z_t = np.random.normal(0, 1)
                    
                    # Evaluate and assign the current value of the short rate
                    short_rate[t_idx+1] = (short_rate[t_idx] + 
                                            (theta_t - a_param*short_rate[t_idx])
                                            *delta_t 
                                            + np.sqrt(delta_t)*sigma*Z_t)
                    
                # Store short rate path in array
                r_t_paths[:,path] = short_rate
                
        elif sim_type.lower() == 'exact':
            # Loop over all short rate paths
            for path in (trange(n_paths) if verbose else range(n_paths)):
                # Loop over all time points of the current path
                for s_idx in range(n_trading_days):
                    # Evaluate the current time point s in years and the array 
                    # index and value in years of the next time point t
                    time_s = s_idx*delta_t
                    t_idx = s_idx + 1
                    time_t = t_idx*delta_t
                    
                    # Compute alpha function values at times t and s
                    alpha_t = alpha_func(a_param, n_annual_trading_days, sigma, 
                                         time_t, time_0_f_curve)
                    alpha_s = alpha_func(a_param, n_annual_trading_days, sigma, 
                                         time_s, time_0_f_curve)
                    
                    # Evaluate the standard deviation of the short rates at 
                    # time t conditional on its value at time s and the 
                    # standard normal draw at time t
                    sigma_r_t = np.sqrt(one_factor_Hull_White_var(a_param, 
                                        sigma, time_s, time_t))
                    Z_t = np.random.normal(0, 1)
                    
                    # Compute short rate at current time point using Equations 
                    # (3.35) and (3.37) from ref. [1] and Equation (3.45) from 
                    # ref. [2]
                    r_t_paths[t_idx,path] = (np.exp(-a_param*delta_t)
                                              *r_t_paths[s_idx,path] + alpha_t 
                                              - alpha_s*np.exp(-a_param*delta_t)
                                              + sigma_r_t*Z_t)
                    
    elif r_t_process_type.lower() == 'zero-mean':
        if verbose:
            print(f'\nGenerating {n_paths:,} zero-mean processes for simulation' 
                  + ' of the short rates.')
        if sim_type.lower() == 'euler':
            x_t_paths = gen_x_t_paths_Euler_Q(a_param, antithetic,
                                              n_annual_trading_days, n_paths, 
                                              seed, sigma, sim_time, verbose)
            
            # Loop over all short rate paths
            for s_idx in range(n_trading_days):
                # Evaluate the array index and time value in years of the next 
                # time point t
                t_idx = s_idx + 1
                time_t = t_idx*delta_t
                
                # Evaluate the alpha(t) function value as defined in Equation 
                # (3.36) on p. 73 of ref. [1]
                alpha_t = alpha_func(a_param, n_annual_trading_days, sigma, 
                                     time_t, time_0_f_curve)
                
                # Evaluate and assign the current short rate values as the 
                # sum of the zero-mean process value and the alpha function 
                # value
                r_t_paths[t_idx] = (x_t_paths[t_idx] + alpha_t)
                
        elif sim_type.lower() == 'exact':
            x_t_paths = gen_x_t_paths_exact_Q(a_param, n_annual_trading_days, 
                                              n_paths, seed, sigma, sim_time, 
                                              verbose)
            
            # Loop over all short rate paths
            for path in (trange(n_paths) if verbose else range(n_paths)):
                # Loop over all time points of the current path
                for s_idx in range(n_trading_days):
                    # Evaluate the current time point s in years and the array 
                    # index and value in years of the next time point t
                    time_s = s_idx*delta_t
                    t_idx = s_idx + 1
                    time_t = t_idx*delta_t
                    
                    # Compute alpha function values at times t and s
                    alpha_t = alpha_func(a_param, n_annual_trading_days, sigma, 
                                         time_t, time_0_f_curve)
                    
                    # Evaluate the standard deviation of the short rates at 
                    # time t conditional on its value at time s
                    sigma_r_t = np.sqrt(one_factor_Hull_White_var(a_param,
                                        sigma, time_s, time_t))
                    
                    # Evaluate and assign the current value of the short rate
                    r_t_paths[t_idx,path] = (np.exp(-a_param*delta_t)
                                              *x_t_paths[s_idx,path] 
                                              + alpha_t)
    
    if verbose:
        print(f'\n{n_paths:,} short rate paths were generated.')    
    
    if r_t_process_type.lower() == 'zero-mean':
        return r_t_paths, x_t_paths
    
    else:
        return r_t_paths



# Auxiliary functions
def alpha_func(a_param: float,
               n_annual_trading_days: int,
               sigma: float,
               time_t: float,
               time_0_f_curve: list
              ) -> float:
    """
    Info:
        This function returns the output of the alpha(t) function as defined in 
        Equation (3.36) on p. 73 of ref. [1].
    
    Input:
        a_param: the constant mean reversion rate parameter in the 1F-HW short 
            rate model.
        
        n_annual_trading_days: an int specifying the number of trading days 
            per year for the time axis discretization.
        
        sigma: the constant volatility factor in the 1F-HW short rate model.
        
        time_t: the time of evaluation in years.
        
        time_0_f_curve: an ndarray containing the time values of the market 
            time-zero instantaneous forward curve.
        
    Output:
        alpha_t: a float specifying value of the alpha function at the 
            evaluation time t.
    """
    # Evaluate the array index of time t
    t_idx = int(time_t*n_annual_trading_days)
    
    # Check whether array index of time t is within bounds of the time-zero 
    # instantaneous forward rate curve array
    if t_idx > len(time_0_f_curve):
        raise ValueError('evaluation time t is out of scope of the time-zero' 
                         + ' instantaneous forward rate curve.')
    
    # Evaluate the alpha function at time t
    alpha_t = (time_0_f_curve[t_idx] + sigma**2/(2*a_param**2)
                        *(1 - np.exp(-a_param*time_t))**2)
    
    return alpha_t

def eval_discount_factors(n_annual_trading_days: int,
                          r_t_paths: list,
                          time_t: float,
                          time_T: float
                         ) -> list:
    """
    Info: This functions evalutes the stochastic discount factor(s) under the 
        risk-neutral measure using one or multiple short rate process paths on 
        the time interval [t, T] by approximating the stochastic integral using 
        Riemann summation of the realized short rate values.
        
    Input:
        n_annual_trading_days: an int specifying the number of trading days 
            per year for the time axis discretization.
            
        r_t_paths: a 2D ndarray containing the simulated short rate paths along 
        a number of columns corresponding to the number of paths and a number 
        of rows being a discrete short rate time series of length equal to the 
        total number of trading days.
            
        time_t: the time of evaluation in years.

        time_T: the future time of evaluation in years.
            
    Output:
        discount_factors: an ndarray containing the pathwise discount factors.
    """
    # Evaluate the array indices of time_t and time_T and the time difference 
    # delta_T
    t_idx = int(time_t*n_annual_trading_days)
    T_idx = int(time_T*n_annual_trading_days)
    Delta_t = 1/n_annual_trading_days
    
    discount_factors = np.exp(-np.sum(r_t_paths[t_idx:T_idx+1], 
                                      axis=0)*Delta_t)
    
    return discount_factors

def gen_x_t_paths_Euler_Q(a_param: float,
                          antithetic: str,
                          n_annual_trading_days: int,
                          n_paths: int,
                          seed: float,
                          sigma: float,
                          sim_time: float,
                          verbose: bool = False
                         ) -> float:
    """
    Info:
        This function generates zero mean process time series under the 
        risk-neutral measure Q using exact simulation.
        
    Input:
        a_param: the constant mean reversion rate parameter in the 1F-HW short 
            rate model.
        
        antithetic: a bool which if True, sets the zero-mean process  to be 
            simulated using antithetic standard normal random draws. 
            Note: this was only implemented for the shifted zero-mean process 
            simulation using the Euler method.
        
        n_annual_trading_days: an int specifying the number of trading days 
            per year for the time axis discretization.
        
        seed: an int specifying the seed for the randon number generator used 
            for the generation of the zero-mean process paths.
        
        sigma: the constant volatility factor in the 1F-HW short rate model.
        
        sim_time: the simulation time in years.
        
        verbose: a bool which if False blocks the function prints.
        
    Output:
        x_t_paths: a 2D ndarray containing the zero-mean process paths along 
            a number of columns corresponding to the number of paths and a 
            number of rows being a discrete short rate time series of length 
            equal to the total number of trading days.
    """
    # Evaluate the total number of trading days, the spacing delta_t of the 
    # time axis, and the random number seed
    n_trading_days = int(sim_time*n_annual_trading_days)
    delta_t = sim_time/n_trading_days
    np.random.seed(seed)
    
    # Initialize array for storing of the zero mean process paths
    x_t_paths = np.zeros((n_trading_days+1,n_paths))
    
    if antithetic:
        # Iterate over all zero mean process paths
        for path in (trange(0, n_paths, 2) if verbose 
                     else range(0, n_paths, 2)):
            # Loop over all time points of the current path
            for s_idx in range(n_trading_days):
                # Evaluate the array index value in years of the next time 
                # point t
                t_idx = s_idx + 1
                
                # Evaluate the standard normal draw at time t
                Z_t = np.random.normal(0, 1)
                Z_t_antithetic = -Z_t
                
                # Evaluate and assign the current value of the zero-mean 
                # process
                x_t_paths[t_idx,path] = (x_t_paths[s_idx,path] 
                                         - a_param*x_t_paths[s_idx,path]*delta_t 
                                         + sigma*np.sqrt(delta_t)*Z_t)
                # Assign the current value of the antithetic path
                x_t_paths[t_idx,path+1] = (x_t_paths[s_idx,path+1] 
                                         - a_param*x_t_paths[s_idx,path+1]*delta_t 
                                         + sigma*np.sqrt(delta_t)*Z_t_antithetic)
        
    else:
        # Iterate over all zero mean process paths
        for path in (trange(n_paths) if verbose else range(n_paths)):
            # Loop over all time points of the current path
            for s_idx in range(n_trading_days):
                # Evaluate the array index value in years of the next time 
                # point t
                t_idx = s_idx + 1
                
                # Evaluate the standard normal draw at time t
                Z_t = np.random.normal(0, 1)
                
                # Evaluate and assign the current value of the zero-mean 
                # process
                x_t_paths[t_idx,path] = (x_t_paths[s_idx,path] 
                                         - a_param*x_t_paths[s_idx,path]*delta_t 
                                         + sigma*np.sqrt(delta_t)*Z_t)
    
    
    return x_t_paths

def gen_x_t_paths_exact_Q(a_param: float,
                          n_annual_trading_days: int,
                          n_paths: int,
                          seed: float,
                          sigma: float,
                          sim_time: float,
                          verbose: bool = False
                         ) -> float:
    """
    Info:
        This function generates zero mean process time series under the 
        risk-neutral measure Q using exact simulation.
        
    Input:
        a_param: the constant mean reversion rate parameter in the 1F-HW short 
            rate model.
        
        n_annual_trading_days: an int specifying the number of trading days 
            per year for the time axis discretization.
        
        seed: an int specifying the seed for the randon number generator used 
            for the generation of the zero-mean process paths.
        
        sigma: the constant volatility factor in the 1F-HW short rate model.
        
        sim_time: the simulation time in years.
        
        verbose: a bool which if False blocks the function prints.
        
    Output:
        x_t_paths: a 2D ndarray containing the zero-mean process paths along 
            a number of columns corresponding to the number of paths and a 
            number of rows being a discrete short rate time series of length 
            equal to the total number of trading days.
    """
    # Evaluate the total number of trading days, the spacing delta_t of the 
    # time axis, and the random number seed
    n_trading_days = int(sim_time*n_annual_trading_days)
    delta_t = sim_time/n_trading_days
    np.random.seed(seed)
    
    # Initialize array for storing of the zero mean process paths
    x_t_paths = np.zeros((n_trading_days+1,n_paths))
    
    # Iterate over all zero mean process paths
    for path in (trange(n_paths) if verbose else range(n_paths)):
        # Loop over all time points of the current path
        for s_idx in range(n_trading_days):
            # Evaluate the current time point s in years and the array index 
            # and value in years of the next time point t
            time_s = s_idx*delta_t
            t_idx = s_idx + 1
            time_t = t_idx*delta_t
            
            # Evaluate the standard deviation of the zero-mean process at time 
            # t conditional on its value at time s
            sigma_x_t = np.sqrt(one_factor_Hull_White_var(a_param, sigma, 
                                                          time_s, time_t))
            
            # Evaluate the standard normal draw at time t
            Z_t = np.random.normal(0, 1)
            
            # Evaluate and assign the current value of the zero-mean process
            x_t_paths[t_idx,path] = (np.exp(-a_param*delta_t)
                                     *x_t_paths[s_idx,path] 
                                     + sigma_x_t*Z_t)
    
    return x_t_paths

def gen_x_t_paths_exact_QT(a_param: float,
                           n_annual_trading_days: int,
                           n_paths: int,
                           sigma: float,
                           sim_time: float,
                           verbose: bool = False
                          ) -> float:
    """
    Info:
        This function generates zero mean process time series under the 
        T-forward risk-adjusted measure Q^T using exact simulation.
        
    Input:
        a_param: the constant mean reversion rate parameter in the 1F-HW short 
            rate model.
        
        n_annual_trading_days: an int specifying the number of trading days 
            per year for the time axis discretization.
        
        n_paths: an int specifying the number of simulated short rate paths.
        
        sigma: the constant volatility factor in the 1F-HW short rate model.
        
        sim_time: the simulation time in years.
        
        verbose: a bool which if False blocks the function prints.
        
    Output:
        x_t_paths: a 2D ndarray containing the zero-mean process paths along 
            a number of columns corresponding to the number of paths and a 
            number of rows being a discrete short rate time series of length 
            equal to the total number of trading days.
    """
    # Evaluate the total number of trading days, the spacing delta_t of the 
    # time axis, and the upper bound time_T of the time interval that is 
    # spanned by the simply-compounded forward rate under the T-forward measure
    n_trading_days = int(sim_time*n_annual_trading_days)
    delta_t = sim_time/n_trading_days
    time_T = sim_time
    
    # Initialize array for storing of the zero mean process paths
    x_t_array = np.zeros((n_trading_days+1,n_paths))
    
    # Iterate over all zero mean process paths
    for path in (trange(n_paths) if verbose else range(n_paths)):
        # Loop over all time points of the current path
        for s_idx in range(n_trading_days):
            # Evaluate the time values in years at times t and s
            time_s = s_idx*delta_t
            t_idx = s_idx + 1
            time_t = t_idx*delta_t
            
            # Evaluate the M function value for times s and t using the 
            # Equation at the top of p. 76 of ref. [1]
            MT_s_t = MT_func(a_param, sigma, time_s, time_t, time_T)

            # Evaluate the standard deviation of the zero-mean process at time 
            # t conditional on its value at time s
            sigma_x_t = np.sqrt(one_factor_Hull_White_var(a_param, sigma, 
                                                          time_s, time_t))
            
            # Evaluate the standard normal draw at time t
            Z_t = np.random.normal(0, 1)
            
            # Evaluate the zero-mean process at the current time step using the 
            # bottom equation on p. 75 of ref. [1]
            x_t_array[t_idx,path] = (np.exp(-a_param*delta_t)
                                     *x_t_array[s_idx,path] - MT_s_t 
                                     + sigma_x_t*Z_t)
            
    return x_t_array

def MT_func(a_param: float,
            sigma: float,
            time_s: float,
            time_t: float,
            time_T: float
           ) -> float:
    """
    Info:
        This function returns the output of the M^T(s, t) function as defined 
        in the top Equation on p. 76 of ref. [1].
    
    Input:
        a_param: the constant mean reversion rate parameter in the 1F-HW model.
        
        sigma: the constant volatility factor of the short rates.
        
        time_s: the previous time in years prior to the current time of 
            evaluation.
            
        time_t: the time of evaluation in years.
        
        time_T: the time in years in which the time interval ends that is 
            spanned by the relevant simply-compounded forward rate under the 
            T-forward measure.
    
    Output:
        MT_s_t: a float specifying the value of the M^T function at the 
        evaluation time t.
    """
    # Check whether time_t is within the bounds of the time interval spanned 
    # by the simply-compounded forward rate
    if time_t > time_T:
        raise ValueError('time_t may not be greater than time_T.')
        
    # Evaluate M^T function value at times s and t using the Equation at the 
    # top of p. 76 from ref. [1]
    MT_s_t = (sigma**2/a_param**2*(1 - np.exp(-a_param*(time_t - time_s))) 
             - sigma**2/(2*a_param**2)*(np.exp(-a_param*(time_T - time_t)) 
                                        - np.exp(-a_param*(time_T + time_t 
                                                           - 2*time_s))))
    
    return MT_s_t

def one_factor_Hull_White_exp_Q(a_param: float,
                                n_annual_trading_days: int,
                                r_t_paths: list,
                                sigma: float,
                                time_s: float,
                                time_t: float,
                                time_0_f_curve: list
                               ) -> float:
    """
    Info:
        This function evaluates the expected mean value of the short rate 
        distribution in the one-factor Hull & White (1F-HW) short rate model 
        under the risk-neutral measure Q.
        
    Input:
        a_param: the constant mean reversion rate parameter in the 1F-HW model.
        
        n_annual_trading_days: the number of trading days in a year.
        
        r_t_paths: a 2D ndarray containing previously simulated short rate 
        paths along a number of columns corresponding to the number of paths 
        and a number of rows being a discrete short rate time series of length 
        equal to the total number of trading days.
        
        sigma: the constant volatility factor of the short rates.
        
        time_s: the previous time point whose short rate variance the current 
            short rate variance is conditioned on.
            
        time_t: the current time point for which the short rate variance is 
            evaluated.
            
        time_0_f_curve: the time-zero instantaneous forward rate curve.
        
    Output:
        expectation: a float specifying the expected mean value of the short 
        rate distribution at the current time point t conditioned on the 
        expected variance of the previous time point s.
    """
    # Evaluate the array index of the previous time s
    s_idx = int(time_s*n_annual_trading_days)
    
    # Evaluate the mean short rate value at time s
    if len(r_t_paths.shape) == 2:
        r_s = np.mean(r_t_paths[s_idx])
    else:        
        r_s = r_t_paths
        
    # Evaluate alpha function value at times s and t
    alpha_s = alpha_func(a_param, n_annual_trading_days, sigma, time_s, 
                         time_0_f_curve)
    alpha_t = alpha_func(a_param, n_annual_trading_days, sigma, time_t, 
                         time_0_f_curve)
    
    # Compute the expectation at time t conditional on the value at time s 
    # using Equation (3.37) from ref. [1]
    expectation = (r_s*np.exp(-a_param*(time_t - time_s)) + alpha_t 
                   - alpha_s*np.exp(-a_param*(time_t - time_s)))
    
    return expectation

def one_factor_Hull_White_exp_QT(a_param: float,
                                 n_annual_trading_days: int,
                                 sigma: float,
                                 time_s: float,
                                 time_t: float,
                                 time_T: float,
                                 time_0_f_curve: list,
                                 x_t_paths: list
                                ) -> float:
    """
    Info:
        This function evaluates the expected mean value of the short rate 
        distribution in the one-factor Hull & White (1F-HW) short rate model 
        under the T-forward risk-adjusted measure Q^T.
        
    Input:
        a_param: the constant mean reversion rate parameter in the 1F-HW short 
            rate model.
        
        n_annual_trading_days: an int specifying the number of trading days 
            per year for the time axis discretization.
            
        sigma: the constant volatility factor in the 1F-HW short rate model.
        
        time_s: the previous time point whose short rate variance the current 
            short rate variance is conditioned on.
            
        time_t: the current time point for which the short rate variance is 
            evaluated.
            
        time_T: the time in years in which the time interval ends that is 
            spanned by the relevant simply-compounded forward rate under the 
            T-forward measure.
            
        time_0_f_curve: the time-zero instantaneous forward rate curve.
        
        x_t_paths: a 2D ndarray containing previously simulated zero-mean 
        process paths along a number of columns corresponding to the number of 
        paths and a number of rows being a discrete short rate time series of 
        length equal to the total number of trading days.
        
    Output:
        expectation: a float specifying expected mean value of the short rate 
        distribution at the current time point t conditioned on the expected 
        variance of the previous time point s.
    """
    # Evaluate the array index of the previous time s
    s_idx = int(time_s*n_annual_trading_days)
    
    # Evaluate the mean short rate value at time s
    if len(x_t_paths.shape) == 2:
        x_s = np.mean(x_t_paths[s_idx])
    else:        
        x_s = x_t_paths
        
    # Evaluate alpha function value at times s and t
    MT = (sigma**2/(a_param**2)*(1 - np.exp(-a_param*(time_t - time_s))) 
          - sigma**2/(2*a_param**2)*(np.exp(-a_param*(time_T - time_t)) 
                                     - np.exp(-a_param*(time_T + time_t 
                                                        - 2*time_s))))
    alpha_t = alpha_func(a_param, n_annual_trading_days, sigma, time_t, 
                         time_0_f_curve)
    
    # Compute the expectation at time t conditional on the value at time s 
    # using Equation (3.37) from ref. [1]
    expectation = (x_s*np.exp(-a_param*(time_t - time_s)) - MT + alpha_t)
    
    return expectation

def one_factor_Hull_White_exp_zero_proc_QT(a_param: float,
                                           n_annual_trading_days: int,
                                           sigma: float,
                                           time_s: float,
                                           time_t: float,
                                           time_T: float,
                                           time_0_f_curve: list,
                                           x_t_paths: list,
                                          ) -> float:
    """
    Info:
        This function evaluates the expected mean value of the short rate 
        distribution in the one-factor Hull & White (1F-HW) short rate model 
        under the T-forward risk-adjusted measure Q^T using the zero-mean 
        process.
        
    Input:
        a_param: the constant mean reversion rate parameter in the 1F-HW short 
            rate model.
        
        n_annual_trading_days: an int specifying the number of trading days 
            per year for the time axis discretization.
            
        sigma: the constant volatility factor in the 1F-HW short rate model.
        
        time_s: the previous time point whose short rate variance the current 
            short rate variance is conditioned on.
            
        time_t: the current time point for which the short rate variance is 
            evaluated.
            
        time_T: the time in years in which the time interval ends that is 
            spanned by the relevant simply-compounded forward rate under the 
            T-forward measure.
            
        time_0_f_curve: the time-zero instantaneous forward rate curve.
        
        x_t_paths: a 2D ndarray containing previously simulated zero-mean 
        process paths along a number of columns corresponding to the number of 
        paths and a number of rows being a discrete short rate time series of 
        length equal to the total number of trading days.
            
    Output:
        expectation: a float specifying the expected mean value of the 
        zero-mean process distribution at the current time point t conditioned 
        on the expected variance of the previous time point s.
    """
    # Evaluate the mean short rate value at time s
    if len(x_t_paths.shape) == 2:
        # Evaluate the array index of the previous time point s
        s_idx = int(time_s*n_annual_trading_days)
        
        # Assign the zero-mean process value at time s
        x_s = np.mean(x_t_paths[s_idx])
    else:        
        x_s = x_t_paths
        
    # Evaluate the M function value for times s and t using the 
    # Equation at the top of p. 76 of ref. [1]
    MT_s_t = MT_func(a_param, sigma, time_s, time_t, time_T)
    
    # Evaluate alpha function value at times s and t
    alpha_t = alpha_func(a_param, n_annual_trading_days, sigma, time_t, 
                         time_0_f_curve)
    
    # Compute the expectation of the short rate at time t conditional on its 
    # value at time s using the expectation Equation at the top of p. 76 from 
    # ref. [1]. Note that the subtraction of the MT_s_t term as displayed in 
    # the Equation was already accounted for in the zero-mean process path 
    # generation function and thus also in the input_x_t_paths array
    expectation = (x_s*np.exp(-a_param*(time_t - time_s)) - MT_s_t + alpha_t)
    
    return expectation
    
def one_factor_Hull_White_var(a_param: float,
                              sigma: float,
                              time_s: float,
                              time_t: float
                             ) -> float:
    """
    Info:
        This function evaluates the expected variance of the short rate 
        distribution in the one-factor Hull & White (1F-HW) short rate model.
        
    Input:
        a_param: the constant mean reversion rate parameter in the 1F-HW short 
            rate model.
        
        sigma: the constant volatility factor in the 1F-HW short rate model.
        
        time_s: the previous time point whose short rate variance the current 
            short rate variance is conditioned on.
            
        time_t: the current time point for which the short rate variance is 
            evaluated.
        
    Output:
        variance: a float specifying the expected variance of the short rate 
        distribution at the current time point t conditioned on the expected 
        variance of the previous time point s.
    """
    # Compute the variance at time t conditional on the value at time s using 
    # Equation (3.37) from ref. [1]
    variance = sigma**2/(2*a_param)*(1 - np.exp(-2*a_param*(time_t - time_s)))
        
    return variance