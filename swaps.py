# -*- coding: utf-8 -*-

# This module provides functions for the valuation of interest rate swaps and 
# their auxiliary functions under the one-factor Hull-White short rate model.
# ...

# References:
#     [1] Brigo, D., & Mercurio, F. (2006). Interest Rate Models - Theory and 
#         Practice: With Smile, Inflation, and Credit. Springer, Berlin, 
#         Heidelberg. doi: 10.1007/978-3-540-34604-3

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
from interest_rate_functions import (eval_annuity_terms,
                                     eval_simply_comp_forward_rate,
                                     eval_swap_rate)
from plotting_functions import (plot_forward_start_swap_timeline, 
                                plot_time_series)

# Function definitions
def eval_moneyness_adjusted_fixed_rate(moneyness: float,
                                       n_annual_trading_days: int,
                                       swap_type: str,
                                       tenor_structure: list,
                                       time_0_P_curve: list):
    """
    Info: This function computes the fixed rate that corresponds to the 
        time-zero swap rate of an (underlying) interest rate swap at the 
        specified level of moneyness.
        
    Inputs:
        moneyness: a float specifying the level of moneyness e.g. 1.0 equals 
            100% moneyness.
            
        n_annual_trading_days: an int specifying the number of trading days 
            per year for the time axis discretization.
            
        swap_type: a str specifying whether the (underlying) interest rate swap 
            is a payer or receiver swap.
            
        tenor_structure: a list containing the (underlying) forward swap's 
            starting date as the first entry and the payment dates as the 
            remaining entries.
            
        time_0_P_curve: an ndarray containing the time values of the market 
            time-zero zero-bond curve.
        
    Output:
        fixed_rate: a float specifying the moneyness-adjusted fixed rate of an 
            (underlying) interest rate swap.
    """
    # Check whether the swap type was specified correctly
    if swap_type.lower() != 'payer' and swap_type.lower() != 'receiver':
        raise ValueError('swap type was not specified correctly. Pass ' 
                         + 'swap_type as "payer" or "receiver".')
        
    # Check whether the level of moneyness was specified correctly
    if moneyness is not None:
        if not isinstance(moneyness, float):
            raise TypeError('moneyness not recognized. Enter as a float ' 
                            + ' specifying the absolute level of moneyness ' 
                            + 'e.g. 1.0 for 100% moneyness.')
        else:
            # Compute the time-zero swap rate of the (underlying) swap and scale 
            # accordingly to find the moneyness-adjusted fixed rate
            time_0_swap_rate = eval_swap_rate(time_0_P_curve, n_annual_trading_days, 
                                              tenor_structure, 0.)
            
            fixed_rate = (time_0_swap_rate/moneyness 
                          if swap_type.lower() == 'payer' 
                          else time_0_swap_rate*moneyness)
    
            return fixed_rate

def price_forward_start_swap(a_param: float,
                             fixed_rate: float,
                             n_annual_trading_days: int,
                             notional: float,
                             payoff_var: str,
                             plot_timeline: bool,
                             r_t_paths: list,
                             sigma: float,
                             swap_type: str,
                             tenor_structure: list,
                             time_t: float,
                             time_0_f_curve: list,
                             time_0_P_curve: list,
                             units_basis_points: bool,
                             x_t_paths: list,
                             verbose: bool = False
                            ):
    """
    Info: This function computes the value of a payer or receiver forward start 
        interest rate swap (IRS) in the one-factor Hull-White (1F-HW) short 
        rate model given the market time-zero zero rate curve, the market 
        time-zero zero-bond curve, and the market time-zero instantaneous 
        forward rate curve.
    
    Input:
        a_param: a float specifying the constant mean reversion rate parameter 
            in the 1F-HW short rate model.
        
        fixed_rate: a float specifying the fixed interest rate used as the 
            strike rate of the (underlying) interest rate swap.
        
        moneyness: a float specifying the level of moneyness e.g. 1.0 equals 
            100% moneyness.
        
        n_annual_trading_days: an int specifying the number of trading days 
            per year for the time axis discretization.
        
        notional: a float specifying the notional amount of the (underlying) 
            interest rate swap.
        
        payoff_var: a str specifying the (underlying) swap payoff function: if 
            'swap', the swap payoff is determined using the forward swap rate; 
            if 'forward', the swap payoff is determined using the simply 
            compounded forward rate.
        
        plot_timeline: a bool specifying whether or not the (underlying) swap 
            timeline is plotted and saved to the local folder.
            
        r_t_paths: a 2D ndarray containing the simulated short rate paths along 
            a number of columns corresponding to the number of paths and a 
            number of rows being a discrete short rate time series of length 
            equal to the total number of trading days.
        
        sigma: a float specifying the constant volatility factor in the 
            1F-HW short rate model.
            
        swap_type: a str specifying the swap type which can be 'payer' or 
            'receiver'.
        
        tenor_structure: a list containing the (underlying) forward swap's 
            starting date as the first entry and the payment dates as the 
            remaining entries.

        time_t: a float specifying the time of evaluation in years.
        
        time_0_f_curve: an ndarray containing the time values of the market 
            time-zero instantaneous forward curve.

        time_0_P_curve: an ndarray containing the time values of the market 
            time-zero zero-bond curve.
        
        units_basis_points: a bool which if True causes the output value to be 
            given in basis points of the notional.
        
        verbose: a bool which if False blocks function prints.
        
        x_t_paths: a 2D ndarray containing the zero-mean process paths along 
            a number of columns corresponding to the number of paths and a 
            number of rows being a discrete short rate time series of length 
            equal to the total number of trading days.
            
    Output: 
        fwd_start_swap_value: an ndarray containing the pathwise time t values 
            of the interest rate swap.
    """
    if verbose:
        print(100*'*')
        print(f'{swap_type.capitalize()} forward-start swap function initialized.\n')
        print(f'Parameters: fixed rate = {fixed_rate}, notional = {notional:,},' 
              + f' maturity = {tenor_structure[-1]}, starting date = {tenor_structure[0]},' 
              + f' payment dates = {tenor_structure[1:]}')
        
    # Check whether swap type was specified correctly and assign the delta 
    # parameter of the payoff function
    if swap_type.lower() == 'payer':
        delta_payoff = 1
    elif swap_type.lower() == 'receiver':
        delta_payoff = -1
    else:
        raise ValueError('forward-start swap type not detected. Enter as "payer" or "receiver"')
        
    # Check whether short rate paths were passed and construct the zero-coupon bond 
    # curves from time t to the final payment date T_M
    if x_t_paths is not None:
        risk_factor_paths = x_t_paths
    elif r_t_paths is not None:
        risk_factor_paths = r_t_paths
    else:
        raise ValueError('no short rate paths or zero-mean process paths were passed.')
        
    # Check whether the final tenor date does not occur after the simulation 
    # time
    if (int(tenor_structure[-1]*n_annual_trading_days) 
        > risk_factor_paths.shape[0]):
        raise ValueError('tenor dates may not exceed length of previously' 
                         + ' simulated short rate paths')
        
    # Construct zero-coupon bond price curve for evaluating the annuities and 
    # swap or forward rates
    tenor = len(tenor_structure)
    time_T_0 = tenor_structure[0]
    time_T_M = tenor_structure[-1]
    ZCBs_t = construct_zero_coupon_bonds_curve(a_param, None, time_T_M, 
                                               n_annual_trading_days, False, 
                                               r_t_paths, sigma, time_t, 
                                               time_0_f_curve, time_0_P_curve, 
                                               x_t_paths, verbose)
    
    # Compute the swap price
    if payoff_var.lower() == 'swap':
        if verbose:
            print(f'Evaluating {swap_type} swap value using forward swap rates...')
            
        # Assign the index values of time_T_0 and time_T_M
        T_0_idx = int(time_T_0*n_annual_trading_days)
        T_M_idx = int(time_T_M*n_annual_trading_days)
        
        # If the evaluation time t is zero, the corresponding zero-coupon bond 
        # values are deterministic and thus equal for all realized short rate 
        # paths. If the evaluation time t is greater than zero, the array index 
        # values of T_0 and T_M need to be adjusted given that the input 
        # zero-coupon bonds curve now has time length shorter by a margin of 
        # time t
        if time_t > 0:
            t_idx = int(time_t*n_annual_trading_days)
            T_0_idx = T_0_idx - t_idx
            T_M_idx = T_M_idx - t_idx
        
        # Evaluate forward swap rate at time t
        fwd_swap_rate = eval_swap_rate(ZCBs_t, n_annual_trading_days, 
                                        tenor_structure, time_t)
        
        if verbose:
            print('fwd_swap_rate: ', fwd_swap_rate)
        
        # Evaluate the annuity for the given tenor structure by taking the sum 
        # of the annuity factors
        annuity_terms = eval_annuity_terms(ZCBs_t, n_annual_trading_days, 
                                           tenor_structure, time_t)
        annuity = np.sum(annuity_terms, axis=0)
        
        # Evaluate the forward swap value at time t
        fwd_start_swap_value = (delta_payoff*notional*(ZCBs_t[T_0_idx] 
                                                       - ZCBs_t[T_M_idx] 
                                                       - annuity*fixed_rate))
        
    elif payoff_var.lower() == 'forward':
        if verbose:
            print(f'Evaluating {swap_type} swap value using simply-compounded' 
                  + ' forward rates...')
        # Evaluate the annuity sum terms for each payment date
        annuity_terms = eval_annuity_terms(ZCBs_t, n_annual_trading_days, 
                                           tenor_structure, time_t)
            
        ## Compute the swap value using the second line from Equation (1.24) on 
        ## p. 14 of ref. [1]
        # Evaluate the simply-compounded forward swap rates for each set of 
        # fixing and payment dates of the tenor structure
        if ZCBs_t.ndim > 1:
            simply_comp_fwd_rates = np.zeros((tenor-1, ZCBs_t.shape[1]))
        else:
            simply_comp_fwd_rates = np.zeros(tenor-1)
        
        for count, time_S in enumerate(tenor_structure[1:]):
            time_T = tenor_structure[count]
            
            # Compute simply compounded forward rates using Equation (1.20) on p. 
            # 12 of ref. [1]
            simply_comp_fwd_rates[count] = eval_simply_comp_forward_rate(
                                                        n_annual_trading_days, 
                                                        time_S, time_t, 
                                                        time_T, ZCBs_t)
        
        fwd_start_swap_value = notional*np.sum(annuity_terms*delta_payoff
                                               *(simply_comp_fwd_rates 
                                                 - fixed_rate), axis=0)
    else:
        raise ValueError('swap payoff variable not recognized. Enter as ' 
                         + '"swap" for forward swap rates or "forward" for ' 
                         + 'simply-compounded forward rates.')
        
    # Print swap value
    if verbose:
        if time_t == 0:
            print(f'Time t = {time_t} value of {swap_type.lower()} forward swap:' 
                  + f' {fwd_start_swap_value}')
        else:
            print(f'Mean time t = {time_t} value of {swap_type.lower()} forward swap:' 
                  + f' {np.mean(fwd_start_swap_value)}')
        
    # Plot swap timeline if desired by user
    if plot_timeline == True:
        if verbose:
            print('\nPlotting mean swap timeline...')
        if time_t > 0:
            plot_forward_start_swap_timeline(fixed_rate, 
                                    np.mean(simply_comp_fwd_rates, axis=0), 
                                    swap_type, tenor_structure)
        else:
            plot_forward_start_swap_timeline(fixed_rate, 
                                    simply_comp_fwd_rates, 
                                    swap_type, tenor_structure)
        
    if verbose:
        print(f'\n{swap_type.capitalize()} forward-start swap function terminated.')
        print(100*'*', '\n')
        
    return (fwd_start_swap_value*10**4/notional if units_basis_points 
            else fwd_start_swap_value)