# -*- coding: utf-8 -*-

# This module provides functions for the valuation of European swaptions using 
# the one-factor Hull-White short rate model. ...

# References:
#     [1] Brigo, D., & Mercurio, F. (2006). Interest Rate Models - Theory and 
#         Practice: With Smile, Inflation, and Credit. Springer, Berlin, 
#         Heidelberg. doi: 10.1007/978-3-540-34604-3
#     [2] Glasserman, P. (2003). Monte Carlo Methods in Financial Engineering. 
#         Springer. doi: 10.1007/978-0-387-21617-1
#     [3] Ajda...

# Imports
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.optimize as so
import scipy.stats as st
from tqdm import trange

# Local imports
from bonds_and_bond_options import (A_func, 
                                    B_func, 
                                    construct_zero_coupon_bonds_curve, 
                                    price_zero_coupon_bond, 
                                    price_zero_coupon_bond_option_exact)
from interest_rate_functions import (eval_annuity_terms,
                                     eval_simply_comp_forward_rate,
                                     eval_swap_rate)
from one_factor_Hull_White_model import (eval_discount_factors,
                                         one_factor_Hull_White_exp_Q)
from plotting_functions import (plot_forward_start_swap_timeline, 
                                plot_time_series)
from swaps import (eval_moneyness_adjusted_fixed_rate, 
                   price_forward_start_swap)

# Swaption function definitions
def eval_approx_vol(a_param: float,
                    n_annual_trading_days: int,
                    sigma: float,
                    swaption_type: str,
                    tenor_structure: list,
                    time_t: float,
                    time_0_f_curve: list,
                    time_0_P_curve: list,
                    time_0_rate: float
                   ) -> float:
    """
    Info: This function computes the approximated volatility for use in the 
        simplified Bachelier pricing model of a European swaption as described 
        in ref. [3].
        
    Input: 
        a_param: a float specifying the constant mean reversion rate parameter 
            in the 1F-HW short rate model.
            
        n_annual_trading_days: an int specifying the number of trading days 
            per year for the time axis discretization.
        
        sigma: a float specifying the constant volatility factor in the 
            1F-HW short rate model.
            
        swaption_type: a str specifying the swaption type which can be 
            'payer' or 'receiver'.
            
        tenor_structure: a list containing the (underlying) forward swap's 
            starting date as the first entry and the payment dates as the 
            remaining entries.
            
        time_t: a float specifying the time of evaluation in years.
        
        time_0_f_curve: an ndarray containing the time values of the market 
            time-zero instantaneous forward curve.
            
        time_0_P_curve: an ndarray containing the time values of the market 
            time-zero zero-bond curve.
            
        time_0_rate: the time-zero interest rate on which the market time-zero 
            zero rate curve is based.
            
    Output:
        approx_vol: a float specifying the approximated volatility.
    """
    # Obtain swaption expiry and maturity from tenor structure
    expiry = tenor_structure[0]
    maturity = tenor_structure[-1]
    
    ## Evaluate the approximated, annuity-denominated zero-coupon bond prices 
    ## with maturities given by the swaption expiry and maturity
    # Evaluate the annuity
    annuity_terms = eval_annuity_terms(time_0_P_curve, n_annual_trading_days, 
                                       tenor_structure, time_t)
    annuity = np.sum(annuity_terms, axis=0)
    
    # Evaluate the approximated, annuity-denominated zero-coupon bond price 
    # with maturity equal to the swaption expiry
    ZCB_expiry = price_zero_coupon_bond(a_param, n_annual_trading_days, 
                                        time_0_rate, sigma, 0., expiry, 
                                        time_0_f_curve, 
                                        time_0_P_curve, None)
    ZCB_expiry_annuity_denom = ZCB_expiry/annuity
    
    # Evaluate the approximated, annuity-denominated zero-coupon bond price 
    # with maturity equal to the swaption maturity
    ZCB_maturity = price_zero_coupon_bond(a_param, n_annual_trading_days, 
                                          time_0_rate, sigma, 0., maturity, 
                                          time_0_f_curve, 
                                          time_0_P_curve, None)
    ZCB_maturity_annuity_denom = ZCB_maturity/annuity
    
    ## Evaluate the C_hat term in Equation (1.47) of ref. [3]
    # Evaluate the time-zero swap rate
    time_0_swap_rate = eval_swap_rate(time_0_P_curve, n_annual_trading_days, 
                                      tenor_structure, 0.)
    # Evaluate the sum term containing annuity-denominated zero-coupon bonds 
    # with maturities equaling the swaption payment dates
    sum_term = np.zeros(len(tenor_structure))
    for count, date in enumerate(tenor_structure[1:]):
        tau = date - tenor_structure[count]
        ZCB_date = price_zero_coupon_bond(a_param, n_annual_trading_days, 
                                          time_0_rate, sigma, 0., date, 
                                          time_0_f_curve, 
                                          time_0_P_curve, None)
        ZCB_date_annuity_denom = ZCB_date/annuity
        sum_term[count] = tau*np.exp(-a_param*date)*ZCB_date_annuity_denom
    sum_term = np.sum(sum_term, axis=0)
    
    C_hat = (np.exp(-a_param*expiry)*ZCB_expiry_annuity_denom 
             - np.exp(-a_param*maturity)*ZCB_maturity_annuity_denom 
             - time_0_swap_rate*sum_term)
    
    # Compute the approximate volatility using Equation (1.48) of ref. [3].
    # Note: ref. [3] contains an error in the evaluation of the integral: the 
    # denominator in the square root term should be 2*a_param**3, not 2*a_param
    approx_vol = (sigma*C_hat*np.sqrt((np.exp(2*a_param*expiry) 
                                       - np.exp(2*a_param*time_t))
                                      /(2*a_param**3)))
    
    return approx_vol

def price_European_swaption_exact_Bachelier(a_param: float,
                                            fixed_rate: float,
                                            moneyness: bool,
                                            n_annual_trading_days: int,
                                            notional: float,
                                            r_t_paths: list,
                                            sigma: float,
                                            swaption_type: str,
                                            tenor_structure: list,
                                            time_t: float,
                                            time_0_f_curve: list,
                                            time_0_P_curve: list,
                                            time_0_rate: float,
                                            units_basis_points: bool,
                                            x_t_paths: list
                                           ) -> float:
    """
    Info: This function computes price of a European swaption in the simplified 
        Bachelier pricing model as described in ref. [3].
        
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
            
        r_t_paths: a 2D ndarray containing the simulated short rate paths along 
            a number of columns corresponding to the number of paths and a 
            number of rows being a discrete short rate time series of length 
            equal to the total number of trading days.
        
        sigma: a float specifying the constant volatility factor in the 
            1F-HW short rate model.
            
        swaption_type: a str specifying the swaption type which can be 
            'payer' or 'receiver'.
            
        tenor_structure: a list containing the (underlying) forward swap's 
            starting date as the first entry and the payment dates as the 
            remaining entries.
            
        time_t: a float specifying the time of evaluation in years.
        
        time_0_f_curve: an ndarray containing the time values of the market 
            time-zero instantaneous forward curve.
            
        time_0_P_curve: an ndarray containing the time values of the market 
            time-zero zero-bond curve.
            
        time_0_rate: the time-zero interest rate on which the market time-zero 
            zero rate curve is based.
            
        units_basis_points: a bool which if True causes the output value to be 
            given in basis points of the notional.
            
        x_t_paths: a 2D ndarray containing the zero-mean process paths along 
            a number of columns corresponding to the number of paths and a 
            number of rows being a discrete short rate time series of length 
            equal to the total number of trading days.
            
    Output:
        swaption_price: an ndarray specifying the pathwise simplified Bachelier 
            prices of the swaption.
    """
    # Determine payoff type
    if swaption_type == 'payer':
        delta_payoff = 1
    elif swaption_type == 'receiver':
        delta_payoff = -1
        
    # If specified, adjust the fixed rate based on the level of moneyness
    if moneyness is not None:
        fixed_rate = eval_moneyness_adjusted_fixed_rate(moneyness, 
                                            n_annual_trading_days, swaption_type, 
                                            tenor_structure, time_0_P_curve)
        
    # Evaluate the swap rate at time_t
    maturity = tenor_structure[-1]
    ZCBs_curve_t = construct_zero_coupon_bonds_curve(a_param, None, 
                                        maturity, n_annual_trading_days, False, 
                                        r_t_paths, sigma, time_t, 
                                        time_0_f_curve, time_0_P_curve, 
                                        x_t_paths)
    
    swap_rate_t = eval_swap_rate(ZCBs_curve_t, n_annual_trading_days, 
                                 tenor_structure, time_t)
    
    # Evaluate the annuity
    annuity_terms = eval_annuity_terms(ZCBs_curve_t, n_annual_trading_days, 
                                       tenor_structure, time_t)
    annuity = np.sum(annuity_terms, axis=0)
    
    
    # Evaluate the approximated volatility
    approx_vol = eval_approx_vol(a_param, n_annual_trading_days, sigma, 
                                 swaption_type, tenor_structure, time_t, 
                                 time_0_f_curve, time_0_P_curve, time_0_rate)
    
    # Compute the swaption price using Equation (1.49) of ref. [3]
    swaption_price = (notional*annuity*(delta_payoff*(swap_rate_t - fixed_rate)
                               *st.norm.cdf(delta_payoff*(swap_rate_t 
                                                          - fixed_rate)
                                            /approx_vol) 
                               + approx_vol
                               *st.norm.pdf(delta_payoff*(fixed_rate 
                                                          - swap_rate_t)
                                            /approx_vol)))
    
    return (swaption_price*10**4/notional if units_basis_points 
            else swaption_price)

def price_European_swaption_exact_Jamshidian(a_param: float,
                                             fixed_rate: float,
                                             moneyness: bool,
                                             n_annual_trading_days: int,
                                             notional: float,
                                             sigma: float,
                                             swaption_type: str,
                                             tenor_structure: list,
                                             time_t: float,
                                             time_0_f_curve: list,
                                             time_0_P_curve: list,
                                             time_0_rate: float,
                                             units_basis_points: bool
                                            ) -> float:
    """
    Info: 
        This function computes the exact price of a European payer of receiver 
        forwad interest rate swaption using Jamshidian's decomposition.
    
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
            
        sigma: a float specifying the constant volatility factor in the 
            1F-HW short rate model.
            
        swaption_type: a str specifying the swaption type which can be 
            'payer' or 'receiver'.
            
        tenor_structure: a list containing the (underlying) forward swap's 
            starting date as the first entry and the payment dates as the 
            remaining entries.
        
        time_t: a float specifying the time of evaluation in years.
        
        time_0_f_curve: an ndarray containing the time values of the market 
            time-zero instantaneous forward curve.
            
        time_0_P_curve: an ndarray containing the time values of the market 
            time-zero zero-bond curve.
        
        time_0_rate: the time-zero interest rate on which the market time-zero 
            zero rate curve is based.
            
        units_basis_points: a bool which if True causes the output value to be 
            given in basis points of the notional.
            
    Output: 
        swaption_price: a float specifying the exact Jamshidian price of the 
            European swaption at time t.
    """
    # Determine main parameters
    expiry = tenor_structure[0]
    maturity = tenor_structure[-1]
    tenor = len(tenor_structure)
    
    # Check the swaption type and determine the corresponding option type of 
    # the portfolio of zero-coupon bond options used in Jamshidian's 
    # decomposition
    if swaption_type.lower() == 'payer':
        ZBO_type = 'put'
    elif swaption_type.lower() == 'receiver':
        ZBO_type = 'call'
    else:
        raise ValueError('swaption type must be "payer" or "receiver".')
        
    if moneyness is not None:
        time_0_swap_rate = eval_swap_rate(time_0_P_curve, 
                                          n_annual_trading_days, 
                                          tenor_structure, 0.)
        fixed_rate = (time_0_swap_rate/moneyness 
                      if swaption_type.lower() == 'payer' 
                      else time_0_swap_rate*moneyness)
        
    ## Evaluate the cashflows at the payment times
    cashflows = np.zeros(tenor-1)
    
    for count, time_T_i in enumerate(tenor_structure[1:]):
        cashflows[count] = fixed_rate*(time_T_i - tenor_structure[count])

    # Adjust final cashflow as given on p. 77 of ref. [1]
    cashflows[-1] += 1.
    
    ## Initialize the valuation function for the portfolio of zero-coupon bonds 
    # for the root-finding algorithm
    def func(x):
        ZCBs = np.zeros(tenor-1)
        
        for count, time_T_i in enumerate(tenor_structure[1:]):
            # Evaluate the zero-coupon bond with maturity T_i
            ZCBs[count] = price_zero_coupon_bond(a_param, n_annual_trading_days, 
                                                 x, sigma, expiry, 
                                                 time_T_i, time_0_f_curve, 
                                                 time_0_P_curve, None)
            
        # print(cashflows, ZCBs)
        # print(np.sum(cashflows*ZCBs))
        
        # Return the sum over cashflows times zero-coupon bonds such that it 
        # equals 1
        return np.sum(cashflows*ZCBs) - 1
    
    # Use classic Brent's method to find the root of the previously defined 
    # function on the sign-changing interval [-.5, .5]
    r_asterisk = so.brentq(func, -.5, .5)
        
    ## Now that r_asterisk has been obtained, the swaption price can be 
    # evaluated using Equations (3.45) or (3.46), depending on the swaption 
    # type, on p. 77-78 of ref. [1].
    # First, for each payment date, price the zero-coupon bond X_i starting at 
    # the swaption maturity and maturing at the payment date with spot rate 
    # r_asterisk, then price the zero-coupon bond option with strike equal to 
    # X_i
    ZBOs = np.zeros(tenor-1)
    maturity_idx = int(maturity*n_annual_trading_days)
    r_t = np.repeat(one_factor_Hull_White_exp_Q(a_param, n_annual_trading_days, 
                                                np.array(time_0_rate), sigma, 
                                                0., time_t, time_0_f_curve), 
                    maturity_idx)
    
    for count, time_T_i in enumerate(tenor_structure[1:]):
        X_i = price_zero_coupon_bond(a_param, n_annual_trading_days, 
                                     r_asterisk, sigma, expiry, 
                                     time_T_i, time_0_f_curve, 
                                     time_0_P_curve, None)
        
        ZBOs[count] = price_zero_coupon_bond_option_exact(a_param, 
                                        time_T_i, expiry, 
                                        n_annual_trading_days, ZBO_type, 
                                        r_t, sigma, X_i, time_t, 
                                        time_0_f_curve, 
                                        time_0_P_curve, 
                                        False, None, None)
        
    # Evaluate the swaption price
    swaption_price = notional*np.sum(cashflows*ZBOs)
    
    return (swaption_price*10**4/notional if units_basis_points 
            else swaption_price)

def price_European_swaption_MC_Q(a_param: float,
                                 experiment_dir: str,
                                 fixed_rate: float,
                                 n_annual_trading_days: int,
                                 notional: float,
                                 payoff_var: str,
                                 plot_timeline: bool,
                                 r_t_paths: list,
                                 sigma: float,
                                 swaption_type: str,
                                 tenor_structure: list,
                                 time_t: float,
                                 time_0_f_curve: list,
                                 time_0_P_curve: list,
                                 units_basis_points: bool,
                                 x_t_paths: list,
                                 verbose: bool = False
                                ) -> list:
    """
    Info: This function computes the Monte Carlo price of a European payer or 
        receiver forward start interest rate swaption in the one-factor 
        Hull-White (1F-HW) short rate model under the risk-neutral measure.
    
    Input:
        a_param: a float specifying the constant mean reversion rate parameter 
            in the 1F-HW short rate model.
        
        experiment_dir: the directory to which the results are saved.
        
        fixed_rate: a float specifying the fixed interest rate used as the 
            strike rate of the (underlying) interest rate swap.
            
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
            
        swaption_type: a str specifying the swaption type which can be 
            'payer' or 'receiver'.
            
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
            
        verbose: a bool which if False blocks the function prints.
        
        x_t_paths: a 2D ndarray containing the zero-mean process paths along 
            a number of columns corresponding to the number of paths and a 
            number of rows being a discrete short rate time series of length 
            equal to the total number of trading days.
        
    Output:
        pathwise_swaption_prices: an ndarray containing the pathwise swaption 
        Monte Carlo prices.
    """
    if verbose:
        print(100*'*')
        print(f'{swaption_type.capitalize()} forward-start swaption function initialized.\n')
    
    # Check whether Data directory located in current directory and if not, 
    # create Data directory
    data_dir = os.path.join(experiment_dir, 'Data\\')

    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    
    # Check whether swaption type was specified correctly and assign the delta 
    # parameter of the payoff function
    if swaption_type.lower() == 'payer':
        delta_payoff = 1
    elif swaption_type.lower() == 'receiver':
        delta_payoff = -1
    else:
        raise ValueError('forward-start swaption type not detected. Enter as "payer" or "receiver"')
        
    # Check whether payoff function variable was specified
    if payoff_var.lower() != 'swap' and payoff_var.lower() != 'forward':
        raise ValueError('no swaption payoff variable was specified.' 
                         + ' Enter as "swap" or "forward".')
    
    # Check whether input short rate paths or zero-mean process paths were 
    # provided
    if r_t_paths is None and x_t_paths is None:
        raise ValueError('no short rate paths or zero-mean process paths ' 
                         + 'were detected.')
    
    if verbose:
        print('\nDetermining forward swap rates using 1F-HW short rate paths...')
    
    # Assign time values of expiry and its array index and of T_M as well 
    # as the array index of the time of evaluation t
    expiry = tenor_structure[0]
    maturity = tenor_structure[-1]
    
    if verbose:
        print('\nConstructing underlying zero-coupon bonds...')
    # Evaluate the stochastic discount factors from time of evaluation t to the 
    # first fixing date expiry for discounting back to evaluation time. The 
    # integral is approximated by summation.
    discount_factors = eval_discount_factors(n_annual_trading_days, 
                                             r_t_paths, time_t, expiry)
    # print('mean discount_factors: ', np.mean(discount_factors))
    
    # Evaluate pathwise zero-coupon bonds from first fixing date expiry to 
    # final payment date T_M for evaluating the swaption payoff function
    pathwise_swaption_prices = discount_factors*np.maximum(
                                            price_forward_start_swap(a_param, 
                                                    fixed_rate, 
                                                    n_annual_trading_days, 
                                                    notional, payoff_var, 
                                                    plot_timeline, r_t_paths, 
                                                    sigma, swaption_type, 
                                                    tenor_structure, expiry, 
                                                    time_0_f_curve, 
                                                    time_0_P_curve, 
                                                    units_basis_points, 
                                                    x_t_paths, verbose), 0.)
        
    if verbose:
        mean_swaption_price = np.mean(pathwise_swaption_prices)
        se_swaption_price = st.sem(pathwise_swaption_prices)
        print(f'Mean European {swaption_type} swaption price at time t={time_t}: {mean_swaption_price}')
        print(f'Standard error {se_swaption_price}:')
    
    # Plot underlying swap timeline if desired by user
    if plot_timeline == True:
        ZCBs_T_0 = construct_zero_coupon_bonds_curve(a_param, experiment_dir, 
                                        maturity, n_annual_trading_days, False, 
                                        r_t_paths, sigma, expiry, time_0_f_curve, 
                                        time_0_P_curve, x_t_paths, verbose)
        simply_comp_fwd_rates = eval_simply_comp_forward_rate(
                                            n_annual_trading_days, 
                                            maturity, time_t, expiry, 
                                            ZCBs_T_0)
        if verbose:
            print('\nPlotting swap timeline...')
            
        if payoff_var.lower() == 'swap':
            pathwise_swap_rates = eval_swap_rate(ZCBs_T_0, n_annual_trading_days, 
                                                  tenor_structure, expiry)
            plot_forward_start_swap_timeline(np.mean(pathwise_swap_rates), 
                                             -delta_payoff*np.mean(simply_comp_fwd_rates, axis=1), 
                                             swaption_type, tenor_structure)
        else:
            
            plot_forward_start_swap_timeline(fixed_rate, 
                                             np.mean(simply_comp_fwd_rates, axis=1), swaption_type, 
                                             tenor_structure)
    
    if verbose:
        print(f'\n{swaption_type.capitalize()} forward-start swaption function terminated.')
        print(100*'*', '\n')
    
    return (pathwise_swaption_prices*10**4/notional if units_basis_points 
            else pathwise_swaption_prices)

def most_expensive_European(a_param: float,
                            Bermudan_price: float,
                            fixed_rate: float,
                            moneyness: bool,
                            n_annual_trading_days: int,
                            notional: float,
                            sigma: float,
                            swaption_type: str,
                            tenor_structure: list,
                            time_t: float,
                            time_0_f_curve: list,
                            time_0_P_curve: list,
                            time_0_rate: float,
                            units_basis_points: bool
                           ) -> float:
    """
    Info: This function computes the exact price of the Most Expensive European 
        (MEE) swaption spanned by the given Bermudan price and tenor structure 
        using Jamshidian's decomposition and returns the MEE gap and the tenor 
        structure of the European swaption.
    
    Input: 
        a_param: a float specifying the constant mean reversion rate parameter 
            in the 1F-HW short rate model.
            
        Bermudan_price: the (mean) price of the Bermudan swaption.
        
        fixed_rate: a float specifying the fixed interest rate used as the 
            strike rate of the (underlying) interest rate swap.
            
        moneyness: a float specifying the level of moneyness e.g. 1.0 equals 
            100% moneyness.
            
        n_annual_trading_days: an int specifying the number of trading days 
            per year for the time axis discretization.
            
        notional: a float specifying the notional amount of the (underlying) 
            interest rate swap.
            
        sigma: a float specifying the constant volatility factor in the 
            1F-HW short rate model.
            
        swaption_type: a str specifying the swaption type which can be 
            'payer' or 'receiver'.
            
        tenor_structure: a list containing the (underlying) forward swap's 
            starting date as the first entry and the payment dates as the 
            remaining entries.
        
        time_t: a float specifying the time of evaluation in years.
        
        time_0_f_curve: an ndarray containing the time values of the market 
            time-zero instantaneous forward curve.
            
        time_0_P_curve: an ndarray containing the time values of the market 
            time-zero zero-bond curve.
        
        time_0_rate: the time-zero interest rate on which the market time-zero 
            zero rate curve is based.
            
        units_basis_points: a bool which if True causes the output value to be 
            given in basis points of the notional.
            
    Output:
        MEE_gap: an ndarray containing the pathwise MEE gaps.
            
        MEE_tenor_structure: the tenor structure of the MEE swaption.
    """
    tenor = len(tenor_structure)
    European_prices_array = np.zeros(tenor-1)
    
    # Compute the prices of all European swaptions spanned by the Bermudan
    for count, __ in enumerate(tenor_structure[:-1]):
        European_prices_array[count] = price_European_swaption_exact_Jamshidian(
                                                a_param, fixed_rate, None, 
                                                n_annual_trading_days, 
                                                notional, sigma, 
                                                swaption_type, 
                                                tenor_structure[count:], 
                                                time_t, time_0_f_curve, 
                                                time_0_P_curve, time_0_rate, 
                                                units_basis_points)
        
    # Determine the index of the MEE swaption
    min_idx = np.argmin(Bermudan_price - European_prices_array)
    
    # Evaluate the MEE gap and tenor structure
    MEE_gap = Bermudan_price - European_prices_array[min_idx]
    MEE_tenor_structure = tenor_structure[min_idx:]
    
    return MEE_gap, MEE_tenor_structure