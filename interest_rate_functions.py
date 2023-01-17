# -*- coding: utf-8 -*-

# This model provides the evaluation functions of various types of interest 
# rates

# References:
#     [1] Brigo, D., & Mercurio, F. (2006). Interest Rate Models - Theory and 
#         Practice: With Smile, Inflation, and Credit. Springer, Berlin, 
#         Heidelberg. doi: 10.1007/978-3-540-34604-3
#     [2] Glasserman, P. (2003). Monte Carlo Methods in Financial Engineering. 
#         Springer. doi: 10.1007/978-0-387-21617-1

# Imports
from datetime import datetime
import numpy as np
import os
from tqdm import trange

# Local imports
from bonds_and_bond_options import (A_func, 
                                    B_func, 
                                    construct_zero_coupon_bonds_curve, 
                                    price_zero_coupon_bond)
from data_functions import write_Parquet_data
from plotting_functions import (plot_forward_start_swap_timeline, 
                                plot_time_series)

def eval_cont_comp_spot_rate(input_zero_coupon_bonds_curve: list,
                             n_annual_trading_days: int,
                             time_t: float, 
                             time_T: float
                            ):
    """
    Info: This function evalutes the value(s) of the continously-compounded 
        spot interest rate R(t,T) as defined in Equation (1.5) in ref. [1].
        
    Input: 
        input_zero_coupon_bonds_curve: an ndarray containing the zero-coupon 
            bond time prices for which the continuously-compounded spot rate(s) 
            will be evaluated.
            
        n_annual_trading_days: an int specifying the number of trading days 
            per year for the time axis discretization.
            
        time_t: the time of evaluation in years.
        
        time_T: the future time of evaluation in years.
            
    Output:
        R_t_T: an ndarray containing the continuously-compounded spot interest 
            rate(s).
    """
    # Evaluate the time difference parameter of the zero-coupon bond valuation 
    # and maturity times
    tau = time_T - time_t
    
    # Evaluate the array indices of time_t and time_T and adjust them according 
    # to the length of the input zero-coupon bonds curve
    ZCBs_t = input_zero_coupon_bonds_curve
    t_idx = int(time_t*n_annual_trading_days)
    T_idx = int(time_T*n_annual_trading_days) - t_idx
    
    # Evaluate the continuously-compounded spot rate(s)
    R_t_T = -np.log(ZCBs_t[T_idx])/tau
    
    return R_t_T

def eval_simply_comp_forward_rate(n_annual_trading_days: int,
                                  time_S: float,
                                  time_t: float,
                                  time_T: float,
                                  zero_coupon_bonds_curve: list
                                 ):
    """
    Info: This function evaluates the simply-compounded forward rates for one 
        or multiple sets of zero-coupon bond values.
    
    Input:
        n_annual_trading_days: an int specifying the number of trading days 
            per year for the time axis discretization.
        
        time_S: a float specifying the maturity of the simply-compounded 
            forward interest rates.
            
        time_t: a float specifying the time of evaluation in years.
        
        time_T: a float specifying the expiry of the simply-compounded forward 
        interest rates.
            
        zero_coupon_bonds_curve: an array containing the pathwise zero-coupon 
            bond prices for maturities ranging from the evaluation time t to 
            the final payment date T_M.
            
    Output: 
        f_t_T_S: an ndarray containing the continuously-compounded spot 
            interest rate(s).
    """
    # Assign the zero-coupon bond curve from the input array and the tenor
    ZCBs_t = zero_coupon_bonds_curve
        
    # Assign time indices and evaluate Delta_T
    time_t_idx = int(time_t*n_annual_trading_days)
    time_S_idx = int(time_S*n_annual_trading_days) - time_t_idx
    time_T_idx = int(time_T*n_annual_trading_days) - time_t_idx
    Delta_T = time_T - time_S
    
    # Compute simply compounded forward rates using Equation (1.20) on p. 
    # 12 of ref. [1]
    simply_comp_fwd_rates = (1/Delta_T*(ZCBs_t[time_T_idx]/ZCBs_t[time_S_idx] 
                                        - 1))
            
    return simply_comp_fwd_rates

def eval_swap_rate(input_zero_coupon_bonds_curve: list,
                   n_annual_trading_days: int,
                   tenor_structure: list,
                   time_t: float
                  ):
    """
    Info: This function evaluates the forward swap rate for one or multiple 
        sets of zero-coupon bond values and a given tenor structure.
    
    Input:
        input_zero_coupon_bonds_curve: an ndarray containing the zero-coupon 
            bond time prices for which the swap rates will be evaluated.
        
        n_annual_trading_days: an int specifying the number of trading days 
            per year for the time axis discretization.
            
        tenor_structure: a list containing the (underlying) forward swap's 
            starting date as the first entry and the payment dates as the 
            remaining entries.
            
        time_t: a float specifying the time of evaluation in years.
            
    Output: 
        An ndarray containing the forward swap values at time t.
    """
    # Assign the zero-coupon bond curve from the input array, the index 
    # values of T_0 (the first fixing date) and T_M (the last payment 
    # date), and the tenor
    ZCBs_t = input_zero_coupon_bonds_curve
    time_T_0 = tenor_structure[0]
    time_T_M = tenor_structure[-1]
    T_0_idx = int(time_T_0*n_annual_trading_days)
    T_M_idx = int(time_T_M*n_annual_trading_days)
    
    # If the evaluation time t is zero, the corresponding zero-coupon bond 
    # values are deterministic and thus equal for all realized short rate 
    # paths. If the evaluation time t is greater than zero, the array index 
    # values of T_0 and T_M need to be adjusted given that the input 
    # zero-coupon bonds curve now has time length shorter by a margin of time t
    time_t_idx = int(time_t*n_annual_trading_days)
    T_0_idx = T_0_idx - time_t_idx
    T_M_idx = T_M_idx - time_t_idx
        
    # Evaluate the numerator of the forward swap rates at time t as in Equation 
    # (1.25) on p. 15 of ref. [1]. Note that Equation (1.25) is the specific 
    # equation for a receiver swap and so has already been multiplied by 
    # delta_payoff = -1
    fwd_swap_rates_numerator = ZCBs_t[T_0_idx] - ZCBs_t[T_M_idx]
    
    # Evaluate the annuity, i.e., the denominator of Equation (1.25) on p. 15 
    # of ref. [1]
    annuities_terms = eval_annuity_terms(ZCBs_t, n_annual_trading_days, 
                                         tenor_structure, time_t)
    
    annuities = np.sum(annuities_terms, axis=0)
    
    # Evaluate and return forward swap rate at time t by dividing the numerator 
    # by the annuity as in Equation (1.25) on p. 15 of ref. [1]
    fwd_swap_rates = fwd_swap_rates_numerator/annuities
    
    return fwd_swap_rates

def eval_annuity_terms(input_zero_coupon_bonds_curve: list,
                       n_annual_trading_days: int,
                       tenor_structure: list,
                       time_t: float
                      ):
    """
    Info: This function evaluates the annuity summand terms for one or 
        multiple sets of zero-coupon bond values.
    
    Input:
        input_zero_coupon_bonds_curve: an ndarray containing the zero-coupon 
            bond time prices for which the annuity terms will be evaluated.
        
        n_annual_trading_days: an int specifying the number of trading days 
            per year for the time axis discretization.
        
        tenor_structure: a list containing the (underlying) forward swap's 
            starting date as the first entry and the payment dates as the 
            remaining entries.
            
        time_t: a float specifying the time of evaluation in years.
            
    Output: 
        annuity_terms: an ndarray containing the pathwise annuity summand 
        terms.
    """
    # Check if zero-coupon bonds curve starting at time t was passed
    if input_zero_coupon_bonds_curve is None:
        raise ValueError('No input zero-coupon bonds curve was passed.')
    
    # Assign the zero-coupon bond curve from the input array and the tenor
    ZCBs_t = input_zero_coupon_bonds_curve
    tenor = len(tenor_structure)
    
    # If the evaluation time t is zero, the corresponding zero-coupon bond 
    # values are deterministic and thus equal for all realized short rate 
    # paths with the same principle applying to the derived annuities. If the 
    # evaluation time t is greater than zero, the corresponding zero-coupon 
    # bond values are stochastic and thus not equal for all realized short rate 
    # paths and therefore the annuities may differ for each path. As a result, 
    # when the evaluation time t is zero, the values from one zero-coupon bond 
    # curve are used to save computation time.
    
    # Initialize array for storing the simply compounded forward rates at time 
    # t and evaluate the array index value of time t
    if ZCBs_t.ndim > 1:
        annuity_terms = np.zeros((tenor-1, ZCBs_t.shape[1]))
    else:
        annuity_terms = np.zeros(tenor-1)
        
    time_t_idx = int(time_t*n_annual_trading_days)
        
    for count, time_T_i in enumerate(tenor_structure[1:]):
        # Evaluate the array index of the current payment date, the previous 
        # payment date, and the time difference Delta_T between the current and 
        # previous payment date
        T_i_idx = int(time_T_i*n_annual_trading_days) - time_t_idx
        time_T_prev = tenor_structure[count]
        Delta_T = time_T_i - time_T_prev
        
        # Evaluate the current annuity factor, i.e., the current sum term in 
        # the denominator of Equation (1.25) on p. 15 of ref. [1]
        annuity_terms[count] = Delta_T*ZCBs_t[T_i_idx]
    
    return annuity_terms

def interpolate_zero_rate(a_param: float,
                          n_annual_trading_days: float,
                          r_t_paths: list,
                          sigma: float,
                          time_t: float,
                          time_T: float,
                          time_T_after: float,
                          time_T_prev: float,
                          time_0_f_curve: list,
                          time_0_P_curve: list,
                          x_t_paths: list, 
                          verbose: bool = False
                         ) -> float:
    """
    Info: This function interpolates the value of a zero-coupon bond priced at 
        some time_t and ending at some time_T occuring between times 
        time_T_prev and time_T_after using linear interpolation of the 
        continuously-compounded spot rates at these times and the relative time 
        fractions.
        
    Input: 
        a_param: the constant mean reversion rate parameter in the 1F-HW short 
            rate model.
            
        n_annual_trading_days: an int specifying the number of trading days 
            per year for the time axis discretization.
            
        r_t_paths: a 2D ndarray containing the simulated short rate paths along 
            a number of columns corresponding to the number of paths and a 
            number of rows being a discrete short rate time series of length 
            equal to the total number of trading days.
        
        sigma: the constant volatility factor in the 1F-HW short rate model.    
        
        time_t: a float specifying the time of evaluation in years.
        
        time_T: a float specifying the future time T for which the 
            continuously-compounded spot rate will be interpolated.
            
        time_T_after: the future time following time T from which the 
            continuously-compounded spot rate at time T will be interpolated.
            
        time_T_prev: the future time preceding time T from which the 
            continuously-compounded spot rate at time T will be interpolated.
            
        time_0_f_curve: an ndarray containing the time values of the market 
            time-zero instantaneous forward curve.

        time_0_P_curve: an ndarray containing the time values of the market 
            time-zero zero-bond curve.
            
        x_t_paths: a 2D ndarray containing the zero-mean process paths along 
            a number of columns corresponding to the number of paths and a 
            number of rows being a discrete short rate time series of length 
            equal to the total number of trading days.
        
        verbose: a bool which if False blocks the function prints.
        
    Output:
        alpha: a float specifying the interpolation fraction of time T on the 
            interval [time_T_prev, time_T_after]
            
        tau: a float specifying the time difference between time T and time t.
            
        R_tau: an ndarray containing the interpolated continuously-compounded 
            spot rates for tau.
        
    """
    # Check whether time_T occurs on [time_T_prev, time_T_after], and evaluate 
    # the time fractions alpha and tau
    if time_T < time_T_prev or time_T > time_T_after:
        raise ValueError('zero-coupon bond maturity time_T does not occur ' 
                         + 'on the interval of interpolation times ' 
                         + '[time_T_prev, time_T_after]')
    if time_T_prev > time_T_after:
        raise ValueError('the left bound of the interpolation interval ' 
                         + '[time_T_prev, time_T_after] must be smaller than ' 
                         + 'the right bound')
    elif time_t > time_T_prev:
        raise ValueError('zero-coupon bond pricing time time_t may not ' 
                         + 'occur on or after the interpolation interval' 
                         + '[time_T_prev, time_T_after]')
    alpha = (time_T_after - time_T)/(time_T_after - time_T_prev)
    tau = time_T - time_t
    
    ## Linear interpolation
    # Construct the zero-coupon bond curve containing prices of zero-coupon 
    # bonds priced at time_t with maturities between time_t and up to 
    # time_T_after
    ZCB_curve_prev = construct_zero_coupon_bonds_curve(a_param, None, 
                                                time_T_prev, 
                                                n_annual_trading_days, False, 
                                                r_t_paths, sigma, time_t, 
                                                time_0_f_curve, time_0_P_curve, 
                                                x_t_paths, verbose)    
    R_t_T_prev = eval_cont_comp_spot_rate(ZCB_curve_prev, 
                                          n_annual_trading_days, time_t, 
                                          time_T_prev)
        
    ZCB_curve_after = construct_zero_coupon_bonds_curve(a_param, None, 
                                                time_T_after, 
                                                n_annual_trading_days, False, 
                                                r_t_paths, sigma, time_t, 
                                                time_0_f_curve, time_0_P_curve, 
                                                x_t_paths, verbose)    
    
    R_t_T_after = eval_cont_comp_spot_rate(ZCB_curve_after, 
                                           n_annual_trading_days, time_t, 
                                           time_T_after)
        
    R_tau = alpha*R_t_T_prev + (1 - alpha)*R_t_T_after
    
    if verbose:
        # Compare to regular ZCB pricing function
        print(f'Mean ZCB priced using R_tau: {np.mean(np.exp(-tau*R_tau))}')
        time_t_idx = int(time_t*n_annual_trading_days)
        r_t_values = r_t_paths[time_t_idx]
        x_t_values = x_t_paths[time_t_idx]
        ZCB_regular = price_zero_coupon_bond(a_param, n_annual_trading_days, 
                                             r_t_values, sigma, time_t, 
                                             time_T, time_0_f_curve, 
                                             time_0_P_curve, 
                                             x_t_values)
        print(f'Mean ZCB priced using regular function: {np.mean(ZCB_regular)}')
        print(f'Absolute difference: {np.mean(ZCB_regular) - np.mean(np.exp(-tau*R_tau))}')
        
    return alpha, tau, R_tau