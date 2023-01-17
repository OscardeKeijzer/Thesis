# -*- coding: utf-8 -*-

# This module provides functions for the valuation of bonds and bond options, 
# as well as their auxiliary functions, using the one-factor Hull-White short 
# rate model. ...

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
import scipy.optimize as so
import scipy.stats as st
from tqdm import trange

# Local imports
from one_factor_Hull_White_model import eval_discount_factors
from plotting_functions import plot_time_series

# Auxiliary functions
def A_func(a_param: float,
           n_annual_trading_days: int,
           r_t_process_type: str,
           sigma: float,
           time_t: float,
           time_T: float,
           time_0_f_curve: list,
           time_0_P_curve: list
          ) -> float:
    """
    Info:
        This function computes the A(t, T) value as given under Equation (3.39) 
        on p. 75 of ref. [1] in case the short rates were simulated directly or 
        the A(t, T) (= G(t, T)) value as given under Equation (38) on p. 9 of 
        the Sterling paper in case the short rates were simulated using the 
        zero-mean process.
        
    Input:
        a_param: a float specifying the constant mean reversion rate parameter 
            in the 1F-HW short rate model.
            
        n_annual_trading_days: an int specifying the number of trading days 
            per year for the time axis discretization.
        
        r_t_process_type: a str specifying whether the short rates will be 
            simulated directly ('direct') or as a shifted zero-mean process 
            ('zero-mean').
        
        sigma: a float specifying the constant volatility factor in the 
            1F-HW short rate model.
        
        time_t: a float specifying the time of evaluation in years.
        
        time_T: a float specifying the future time of evaluation in years.
            
        time_0_f_curve: the market time-zero instantaneous forward 
            curve.
        
        time_0_P_curve: the market time-zero zero-coupon bond curve.
        
    Output:
        A_t_T: a float specifying the The A(t, T) value as used in computing 
            the price of a zero-coupon bond P(t, T).
    """
    # Evaluate the array index values of t and T
    t_idx = int(time_t*n_annual_trading_days)
    T_idx = int(time_T*n_annual_trading_days)
    
    # Evaluate the B(t, T) function value for the given times t and T
    B_t_T = B_func(a_param, time_t, time_T)
    
    # Evaluate the A(t, T) function value for the given times t and T
    if (r_t_process_type.lower() == 'direct' 
        and time_0_f_curve is not None):
        A_t_T = (time_0_P_curve[T_idx]
                 /time_0_P_curve[t_idx]
                 *np.exp(B_t_T*time_0_f_curve[t_idx] 
                         - sigma**2/(4*a_param)*(1 - np.exp(-2*a_param*time_t))
                         *B_t_T**2))
    elif r_t_process_type.lower() == 'zero-mean':
        A_t_T = (time_0_P_curve[T_idx]/time_0_P_curve[t_idx]
                 *np.exp(-sigma**2/(2*a_param)
                         *B_t_T*(1 - np.exp(-a_param*time_t))
                 *(B_t_T/2*(1 + np.exp(-a_param*time_t)) 
                   + (1 - np.exp(-a_param*time_t))/a_param)))
        
    return A_t_T

def B_func(a_param: float,
           time_t: float,
           time_T: float
          ) -> float:
    """
    Info:
        This function computes the B(t, T) value as given above Equation (3.39) 
        on p. 75 ref. [1]
        
    Input:
        a_param: a float specifying the constant mean reversion rate parameter 
            in the 1F-HW short rate model.
        
        time_t: a float specifying the time of evaluation in years.
        
        time_T: a float specifying the future time of evaluation in years.
            
    Output:
        B_t_T: a float specifying the B(t, T) value as used in computing the 
            price of a zero-coupon bond P(t, T).
    """
    B_t_T = (1 - np.exp(-a_param*(time_T - time_t)))/a_param
    
    return B_t_T

# Bond and bond option valuation functions
def construct_zero_coupon_bonds_curve(a_param: float,
                                      experiment_dir: str,
                                      max_time_T: float,
                                      n_annual_trading_days: int,
                                      plot_curve: bool,
                                      r_t_paths: list,
                                      sigma: float,
                                      time_t: float,
                                      time_0_f_curve: list,
                                      time_0_P_curve: list,
                                      x_t_paths: list,
                                      verbose: bool = False
                                     ) -> list:
    """
    Info: This function constructs a curve for zero-coupon bond prices with 
        maturities ranging from t to T in the one-factor Hull-White (1F-HW) 
        short rate model given the market time-zero zero-coupon curve, 
        the market time-zero zero-bond curve, and the market time-zero 
        instantaneous forward rate curve.
    
    Input:
        a_param: a float specifying the constant mean reversion rate parameter 
            in the 1F-HW short rate model.
        
        experiment_dir: the directory to which the results are saved.
        
        max_time_T: the simulation time in years corresponding to the 
            maximum maturity for which a zero-coupon bond will be priced.
        
        n_annual_trading_days: an int specifying the number of trading days 
            per year for the time axis discretization.
        
        plot_curve: a bool specifying whether or not the resulting curve is 
            plotted and saved to the local folder.
            
        r_t_paths: a 2D ndarray containing the simulated short rate paths along 
            a number of columns corresponding to the number of paths and a 
            number of rows being a discrete short rate time series of length 
            equal to the total number of trading days.
            
        sigma: a float specifying the constant volatility factor in the 
            1F-HW short rate model.
        
        time_t: a float specifying the time of evaluation in years.
        
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
        P_t_T_array: an ndarray containing the pathwise prices P(t, T) for 
            maturities from t to T of a zero-coupon bond under the affine term 
            structure of the 1F-HW model.
    """
    if verbose:
        print(100*'*')
        print('Zero-coupon bond curve function initialized.\n')
    # Store function starting time for timing purposes
    function_starting_time = datetime.now()
    
    # Check whether Data directory located in current directory and if not, 
    # create Data directory
    if experiment_dir is not None:
        data_dir = os.path.join(experiment_dir, 'Data\\')
    
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)
    
    # Check whether the time of evaluation t is not larger than simulation time
    if time_t > max_time_T:
        raise ValueError('time_t may not be greater than max_time_T.')
    
    # Check whether short rate paths or zero-mean paths were passed
    if x_t_paths is not None:
        if len(x_t_paths.shape) > 1:
            n_paths = x_t_paths.shape[1]
        else:
            n_paths = 1
    elif r_t_paths is not None:
        if len(r_t_paths.shape) > 1:
            n_paths = r_t_paths.shape[1]
        else:
            n_paths = 1
    else:
        raise ValueError('pass either short rate paths or zero-mean process paths.')
        
    # Evaluate the number of trading days and the spacing of the time axis 
    # Delta_t
    n_trading_days = int(max_time_T*n_annual_trading_days)
    Delta_t = 1/n_annual_trading_days
    
    # Evaluate the time t price of zero-coupon bonds with maturities ranging 
    # from time t to the simulation time
    t_idx = int(time_t*n_annual_trading_days)
    P_t_T_array = np.zeros((n_trading_days+1, n_paths))
    
    if verbose:
        print(f'Valuing zero-coupon bonds P({time_t}, T) for T in' 
              + f' [{time_t}, {max_time_T}]')
    
    # Construct the curve
    for T_idx in (trange(t_idx, n_trading_days+1) if verbose 
                  else range(t_idx, n_trading_days+1)):
        # Evaluate the array index of the current maturity T in years
        if T_idx%n_annual_trading_days == 0:
            time_T = T_idx//n_annual_trading_days
        else:
            time_T = T_idx*Delta_t
        # Evaluate each zero-coupon bond price P(0, T) with 
        # T = idx/n_annual_trading_days by using the affine term structure 
        # function with realized short rate or zero-mean process values
        if x_t_paths is not None:
            P_t_T_array[T_idx-t_idx] = price_zero_coupon_bond(a_param, 
                                        n_annual_trading_days, None, sigma, 
                                        time_t, time_T, time_0_f_curve, 
                                        time_0_P_curve, x_t_paths[t_idx])
        elif r_t_paths is not None:
            P_t_T_array[T_idx-t_idx] = price_zero_coupon_bond(a_param, 
                                        n_annual_trading_days, 
                                        r_t_paths[t_idx], sigma, time_t, 
                                        time_T, time_0_f_curve, time_0_P_curve, 
                                        None)
                
    # Plot results if desired by user
    if plot_curve == True:
        plot_title = 'Simulated Zero-Coupon Bond Prices Curve'
        x_label = 'Time $T$'
        x_limits = [time_t, max_time_T]
        y_label = f'$P({time_t}, T$)'
        if time_t == 0.:
            y_label = '$P(0, T)$'
        y_limits = None
        
        if verbose:
            print('\nPlotting results...')
        plot_time_series(experiment_dir, n_annual_trading_days, False, 
                         plot_title, P_t_T_array, None, x_label, x_limits, 
                         y_label, y_limits)
        if not time_t == 0:
            plot_time_series(experiment_dir, n_annual_trading_days, True, 
                             plot_title, P_t_T_array, n_x_ticks=None, 
                             x_label=x_label, x_limits=x_limits, 
                             y_label=y_label, y_limits=y_limits)
    
    # Evaluate program runtime
    function_finish_time = datetime.now()
    runtime = (function_finish_time - function_starting_time)
    
    if verbose:
        print(f'\nTotal zero-coupon bond valuation function runtime: {runtime}')
        print('\nZero-coupon bond curve function terminated.')
        print(100*'*', '\n')
        
    return P_t_T_array

def price_coupon_bearing_bond(a_param: float,
                              coupon_dates: list,
                              fixed_rate: float,
                              n_annual_trading_days: int,
                              notional: float,
                              r_t_paths: list,
                              sigma: float,
                              time_t: float,
                              time_T: float,
                              time_0_f_curve: list,
                              time_0_P_curve: list,
                              x_t_paths: list
                             ) -> float:
    """
    Info: This function computes the time t price(s) of one or more 
        coupon-bearing bonds with maturity T in the one-factor Hull-White 
        (1F-HW) short rate model given one or more 1F-HW input short rate 
        values and the time-zero zero-coupon bond and instantaneous forward 
        rate curves.
    
    Input:
        a_param: a float specifying the constant mean reversion rate parameter 
            in the 1F-HW short rate model.
            
        coupon_dates: a list containing the dates on which the underlying 
            bond pays the coupons.
        
        fixed_rate: a float specifying the fixed interest rate used as the 
            strike rate of the (underlying) interest rate swap.
        
        notional: a float specifying the notional of the coupon-bearing bond.
        
        n_annual_trading_days: an int specifying the number of trading days 
            per year for the time axis discretization.
        
        n_paths: an int specifying the number of simulated short rate paths.
            
        r_t_paths: a 2D ndarray containing the simulated short rate paths along 
            a number of columns corresponding to the number of paths and a 
            number of rows being a discrete short rate time series of length 
            equal to the total number of trading days.
        
        sigma: a float specifying the constant volatility factor in the 
            1F-HW short rate model.
        
        time_t: a float specifying the starting time in years of the 
            zero-coupon bond.
        
        time_T: a float specifying the maturity of the coupon-bearing bond.
        
        time_0_f_curve: an ndarray containing the time values of the market 
            time-zero instantaneous forward curve.

        time_0_P_curve: an ndarray containing the time values of the market 
            time-zero zero-bond curve.
        
        x_t_paths: a 2D ndarray containing the zero-mean process paths along 
            a number of columns corresponding to the number of paths and a 
            number of rows being a discrete short rate time series of length 
            equal to the total number of trading days.
        
    Output:
        CB_t: a float specifying the price at time t of the coupon-bearing 
            bond(s) maturing at time T.
    """
    # Check whether short rate paths or zero-mean process paths were passed as 
    # input
    if x_t_paths is not None:
        risk_factor_paths = x_t_paths
    elif r_t_paths is not None and x_t_paths is None:
        risk_factor_paths = r_t_paths
    else:
        raise ValueError('no input short rate or zero mean process values' 
                         + ' were passed.')
    # Check whether payment dates occur on the time interval of the input 
    # zero-mean process paths
    if coupon_dates[-1] > risk_factor_paths.shape[0]/n_annual_trading_days:
        raise ValueError('one or more payment dates do not occur within' 
                         + ' the timeframe of the simulated zero-mean' 
                         + ' process paths.')
            
    # Evaluate the number of paths
    n_paths = np.shape(risk_factor_paths)[1]
        
    # Assign the number of coupon payment dates, the array index value of the 
    # evaluation time t, and initialize arrays for storing the cashflows and 
    # zero-coupon bond values
    n_coupon_dates = len(coupon_dates)
    t_idx = int(time_t*n_annual_trading_days)
    cashflows = np.zeros(n_coupon_dates)
    ZCBs = np.zeros((n_coupon_dates, n_paths))
    
    # Evaluate the summands of the coupon-bearing bonds price Equation 
    # given on p. 15 of ref [1]
    for T_i_idx, time_T_i in enumerate(coupon_dates):
        # Evaluate the cashflows in between the first and final cashflows
        if T_i_idx > 0 and T_i_idx < n_coupon_dates - 1:
            cashflows[T_i_idx] = (notional*(time_T_i - coupon_dates[T_i_idx-1])
                                  *fixed_rate)

        # Evaluate the B(t, T) function value as defined on p. 75 of 
        # ref [1] for T = T_i
        B_t_T = B_func(a_param, time_t, time_T_i)
        
        # Evaluate the A(t, T) function value as defined on p. 75 of 
        # ref [1] for T = T_i
        if x_t_paths is not None and r_t_paths is None:
            A_t_T = A_func(a_param, n_annual_trading_days, 'zero-mean', 
                           sigma, time_t, time_T_i, None, time_0_P_curve)
        elif r_t_paths is not None and x_t_paths is None:
            A_t_T = A_func(a_param, n_annual_trading_days, 'direct', 
                           sigma, time_t, time_T_i, time_0_f_curve, 
                           time_0_P_curve)

        # Evaluate the zero-coupon bond price P(t, T) given the zero-mean 
        # process value(s) at time t for T = T_i
        ZCBs[T_i_idx] = A_t_T*np.exp(-B_t_T*risk_factor_paths)[t_idx]
            
    # Correct the first and final cashflows that were not properly evaluated in 
    # the above loop
    cashflows[0] = notional*(coupon_dates[0] - time_T)*fixed_rate
    cashflows[-1] = (notional
                     *(coupon_dates[-1] - coupon_dates[-2])*fixed_rate)

            
    # Compute the coupon-bearing bond price at time t by evaluating the sum in 
    # the Equation on p. 15 of ref [1]
    ZCBs = np.mean(ZCBs, axis=1)
    CB_t = np.sum(cashflows*ZCBs)
    
    return CB_t

def price_coupon_bearing_bond_option(a_param: float,
                                     coupons: list,
                                     coupon_dates: list,
                                     experiment_dir: str,
                                     fixed_rate: float,
                                     maturity_bond: float,
                                     maturity_option: float,
                                     n_annual_trading_days: int,
                                     notional: float,
                                     option_type: str,
                                     r_t_paths: list,
                                     sigma: float,
                                     strike_price: float,
                                     time_t: float,
                                     time_0_f_curve: list,
                                     time_0_P_curve: list,
                                     zero_coupon_bond: list,
                                     x_t_paths: list,
                                     verbose: bool = False
                                    ) -> float:
    """
    Info: This function computes the exact price of a coupon-bearing bond 
        call or put option in the one-factor Hull-White (1F-HW) short rate 
        model under the T-forward risk-adjusted measure Q^T using Jamshidian's 
        decomposition.
    
    Input:
        a_param: the constant mean reversion rate parameter in the 1F-HW short 
            rate model.
            
        coupons: an array containing the prices of the coupons that the 
            underlying bond pays out on the payment dates.
            
        coupon_dates: an array containing the dates on which the underlying 
            bond pays the coupons.
        
        experiment_dir: the directory to which the results are saved.
        
        fixed_rate: the fixed rate from which the cashflows of the underlying 
            coupon-bearing bond are defined.
        
        maturity_bond: the maturity in years of the zero-coupon bond.
        
        maturity_option: the maturity in years of the option on the 
            zero-coupon bond.

        n_annual_trading_days: the number of trading days per year, used as the 
            time axis discretization.
            
        notional: the notional of the underlying coupon-bearing bond.
            
        option_type: the type of the coupon-bearing bond option, can be 'call' 
            or 'put'.
            
        r_t_paths: previously simulated 1F-HW short rate paths under the 
            T-forward measure Q^T.
        
        sigma: the constant volatility factor of the short rates.
        
        strike_price: the strike price of the coupon-bearing bond option.
        
        time_t: the evaluation time of the option price.
        
        time_0_f_curve: the market time-zero instantaneous 
            forward curve.
        
        time_0_P_curve: the time-zero zero-bond curve from which 
            the time-zero instantaneous forward curve is constructed.
            
        x_t_paths: the current price(s) of the zero-mean process(es).
        
        zero_coupon_bond: the array containing previously priced 
            zero-coupon bonds.
            
    Output:
        option_prices: the pathwise prices of the coupon-bearing bond option.
    """
    if verbose:
        print(100*'*')
        print(f'Coupon-bearing bond {option_type} option valuation function initialized.\n')
    
    # Check whether Data directory located in current directory and if not, 
    # create Data directory
    data_dir = os.path.join(experiment_dir, 'Data\\')

    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
        
    # Check whether payment dates occur on the time interval of the input short 
    # rate or zero-mean process paths; also assign the number of paths
    if x_t_paths is not None:
        risk_factor_paths = x_t_paths
        
        if coupon_dates[-1] > x_t_paths.shape[0]/n_annual_trading_days:
            raise ValueError('one or more payment dates do not occur within' 
                             + ' the timeframe of the simulated zero-mean' 
                             + ' process paths.')
            
        n_paths = x_t_paths.shape[1]
        
    elif r_t_paths is not None and x_t_paths is None:
        risk_factor_paths = r_t_paths
        
        if coupon_dates[-1] > r_t_paths.shape[0]/n_annual_trading_days:
            raise ValueError('one or more payment dates do not occur within' 
                             + ' the timeframe of the simulated short rate' 
                             + ' paths.')
            
        n_paths = r_t_paths.shape[1]
            
    else:
        raise ValueError('no short rate paths or zero-mean process paths' 
                         + ' were passed.')
    
        
    # Check whether evaluation time t does not occur after option maturity
    if time_t > maturity_option:
        raise ValueError('evaluation time time_t may not occur after option maturity.')
    
    # Check whether option type was specified correctly
    if not option_type.lower() == 'call' and not option_type.lower() == 'put':
        raise ValueError('coupon-bearing bond option type not detected.' 
                         + ' Enter as "put" or "call"')
        
    # Evaluate the spot rate r_asterisk for which the coupon-bearing bond price 
    # equals the strike price
    def func(x):
        return (price_coupon_bearing_bond(a_param, coupon_dates, 
                                         n_annual_trading_days, fixed_rate, 
                                         notional, 
                                         np.repeat(x, risk_factor_paths.shape[0]), 
                                         sigma, maturity_option, 
                                         maturity_bond, time_0_f_curve, 
                                         time_0_P_curve, 
                                         None) - strike_price)
        
    r_asterisk = so.fsolve(func, .03)
        
    # Initialize the array for storing the zero-coupon bond option prices that 
    # are used to evaluate the coupon-bearing bond option price using 
    # Jamshidian's decomposition as given in Equation (3.44) on p. 77 of ref. 
    # [1]
    n_coupon_dates = len(coupon_dates)
    ZCB_option_prices = np.zeros((n_coupon_dates, n_paths))
    
    # Evaluate the array index value of the evaluation time t and the option 
    # maturity
    T_idx = int(maturity_option*n_annual_trading_days)
    t_idx = int(time_t*n_annual_trading_days)
    
    # Evaluate the prices of the zero-coupon bond options
    for T_i_idx, time_T_i in enumerate(coupon_dates):
        # Determine the strike price of the zero-coupon bond maturing at 
        # payment date T_i
        # TO-DO: Denote by r∗ the value of the spot rate at time T for which the 
        # coupon-bearing bond price equals the strike and by Xi the time-T value 
        # of a pure-discount bond maturing at Ti when the spot rate is r∗
        X_i = price_zero_coupon_bond(a_param, n_annual_trading_days, 
                                     np.repeat(r_asterisk, risk_factor_paths.shape[0]), 
                                     sigma, time_t, 
                                     time_T_i, time_0_f_curve, 
                                     time_0_P_curve, 
                                     x_t_paths)[T_idx]
        
        ZCB_option_prices[T_i_idx] = price_zero_coupon_bond_option_exact(
                                            a_param, experiment_dir, 
                                            n_annual_trading_days, time_T_i, 
                                            maturity_option, option_type, 
                                            r_t_paths, sigma, X_i, time_t, 
                                            time_0_f_curve, time_0_P_curve, 
                                            zero_coupon_bond, x_t_paths)[t_idx]
    
    # Check whether the coupons were passed as a numpy array and if not convert 
    # the passed list to a numpy array. This is needed for determining the 
    # cashflows by subtracting the strike price from each coupon
    if not type(coupons) == np.ndarray:
        coupons = np.array(coupons)
        
    # Evaluate the cashflows from the coupons and strike price
    cashflows = coupons - strike_price
    
    # Evaluate the coupon-bearing bond option price using the summation in 
    # Equation (3.44) on p. 77 of ref. [1]
    ZCB_option_prices = np.mean(ZCB_option_prices, axis=1)
    option_prices = np.sum(cashflows*ZCB_option_prices)    
    
    if verbose:
        if len(option_prices.shape) == 1:
            print(f'Coupon-bearing bond {option_type.lower()} option price at time'
                  + f' t={time_t}: {option_prices}')
        else:
            print(f'Mean coupon-bearing bond {option_type.lower()} option price' 
                  + f' at time t={time_t}: {np.mean(option_prices)}')
    
    return option_prices

def price_zero_coupon_bond(a_param: float,
                           n_annual_trading_days: float,
                           r_t_values: list,
                           sigma: float,
                           time_t: float,
                           time_T: float,
                           time_0_f_curve: list,
                           time_0_P_curve: list,
                           x_t_values: list
                          ) -> float:
    """
    Info: This function computes the time t price(s) of one or more zero-coupon 
        bonds with maturity T in the one-factor Hull-White (1F-HW) short rate 
        model given one or more 1F-HW input short rate values and the time-zero 
        zero-coupon bond and instantaneous forward rate curves.
    
    Input:
        a_param: a float specifying the constant mean reversion rate parameter 
            in the 1F-HW short rate model.
        
        n_annual_trading_days: an int specifying the number of trading days 
            per year for the time axis discretization.
        
        r_t_values: the current value(s) of the short rate(s).
        
        sigma: a float specifying the constant volatility factor in the 
            1F-HW short rate model.
        
        time_t: the starting time in years of the zero-coupon bond.
        
        time_T: the maturity of the zero-coupon bond.
        
        time_0_f_curve: an ndarray containing the time values of the market 
            time-zero instantaneous forward curve.
            
        time_0_P_curve: an ndarray containing the time values of the market 
            time-zero zero-bond curve.
        
        x_t_values: the current value(s) of the zero-mean process(es).
        
    Output:
        P_t_T: The pathwise prices P(t, T) of zero-coupon bond(s) 
            under the affine term structure of the 1F-HW model.
    """
    # Evaluate the B(t, T) function value as defined on p. 75 of ref [1]
    B_t_T = B_func(a_param, time_t, time_T)
    
    # Evaluate the A(t, T) function value as defined on p. 75 of ref [1] 
    # depending on whether input short rates or zero-mean process values are 
    # used
    if x_t_values is not None:
        A_t_T = A_func(a_param, n_annual_trading_days, 'zero-mean', sigma, 
                       time_t, time_T, None, time_0_P_curve)
        
        # Evaluate the zero-coupon bond price P(t, T) given the zero-mean 
        # process value(s) at time t
        P_t_T = A_t_T*np.exp(-B_t_T*x_t_values)
        
    elif r_t_values is not None and x_t_values is None:
        A_t_T = A_func(a_param, n_annual_trading_days, 'direct', sigma, time_t, 
                       time_T, time_0_f_curve, time_0_P_curve)
        
        # Evaluate the zero-coupon bond price P(t, T) given the short rate 
        # value(s) at time t
        P_t_T = A_t_T*np.exp(-B_t_T*r_t_values)
        
    else:
        raise ValueError('no input short rate or zero mean process values were passed.')
    
    return P_t_T

# Bond option valuation functions
def price_zero_coupon_bond_forward_exact(a_param: float,
                                         maturity_bond: float,
                                         maturity_option: float,
                                         n_annual_trading_days: int,
                                         option_type: str,
                                         r_t_paths: list,
                                         sigma: float,
                                         strike_price: float,
                                         time_t: float,
                                         time_0_f_curve: list,
                                         time_0_P_curve: list,
                                         units_basis_points: bool,
                                         x_t_paths: list,
                                         zero_coupon_bond: list
                                       ) -> float:
    """
    Info: This function computes the exact price of a zero-coupon bond forward 
        in the one-factor Hull-White (1F-HW) short rate model.
    
    Input:
        a_param: the constant mean reversion rate parameter in the 1F-HW short 
            rate model.
        
        maturity_bond: the maturity in years of the underlying zero-coupon bond.
        
        maturity_option: the maturity in years of the option on the 
            zero-coupon bond.
            
        n_annual_trading_days: the number of trading days per year, used as the 
            time axis discretization.
            
        option_type: the type of the zero-coupon bond option, can be 'call' or 
            'put'.
        
        r_t_paths: the time evolution of the simulated 1F-HW short rate 
            paths.
        
        sigma: the constant volatility factor of the short rates.
        
        strike_price: the strike price of the zero-coupon bond option.
        
        time_t: the evaluation time of the option price.
        
        time_0_f_curve: the market time-zero instantaneous 
            forward curve.
        
        time_0_P_curve: the time-zero zero-bond curve from which 
            the time-zero instantaneous forward curve is constructed.
            
        units_basis_points: if True, the price is given in basis points of the 
            notional.
            
        x_t_paths: the time evolution of the zero-mean process(es).
        
        zero_coupon_bond: the array containing previously priced 
            zero-coupon bonds.
            
    Output:
        forward_prices: an ndarray containing the pathwise prices of the 
            zero-coupon bond forward.
    """
    # print(100*'*')
    # print(f'Zero-coupon bond {option_type} option valuation function initialized.\n')
    
    ## Check whether evaluation time t does not occur after option maturity
    if time_t > maturity_option:
        raise ValueError('evaluation time t may not occur after option maturity.')
    else:
        t_idx = int(time_t*n_annual_trading_days)
    
    ## Check whether option type was specified correctly and assign the delta 
    # parameter of the payoff function
    if option_type.lower() == 'call':
        delta_payoff = 1
    elif option_type.lower() == 'put':
        delta_payoff = -1
    else:
        raise ValueError('zero-coupon bond option type not detected.' 
                         + ' Enter as "put" or "call"')
        
    if r_t_paths is None and x_t_paths is None:
        raise ValueError('no short rate paths or zero-mean process paths' 
                          + ' were passed.')
        
    ## Price the zero-coupon bonds P(t, T) and P(t, S) if the latter was not 
    ## provided
    if x_t_paths is not None:
        P_t_T = price_zero_coupon_bond(a_param, n_annual_trading_days, 
                                       None, sigma, time_t, maturity_option, 
                                       time_0_f_curve, time_0_P_curve, 
                                       x_t_paths[t_idx])
        
        if zero_coupon_bond is not None:
            P_t_S = zero_coupon_bond
        else:
            P_t_S = price_zero_coupon_bond(a_param, n_annual_trading_days, 
                                           None, sigma, time_t, maturity_bond, 
                                           time_0_f_curve, time_0_P_curve, 
                                           x_t_paths[t_idx])
            
    elif r_t_paths is not None and x_t_paths is None:
        P_t_T = price_zero_coupon_bond(a_param, n_annual_trading_days, 
                                       r_t_paths[t_idx], sigma, time_t, 
                                       maturity_option, time_0_f_curve, 
                                       time_0_P_curve, None)
        
        if zero_coupon_bond is not None:
            P_t_S = zero_coupon_bond
        else:
            P_t_S = price_zero_coupon_bond(a_param, n_annual_trading_days, 
                                           r_t_paths[t_idx], 
                                           sigma, time_t, maturity_bond, 
                                           time_0_f_curve, 
                                           time_0_P_curve, 
                                           None)
        
    ## Evaluate the option price using Equation...
    forward_prices = delta_payoff*(P_t_S - strike_price*P_t_T)
    
    return forward_prices*10**4 if units_basis_points else forward_prices

def price_zero_coupon_bond_option_exact(a_param: float,
                                        maturity_bond: float,
                                        maturity_option: float,
                                        n_annual_trading_days: int,
                                        option_type: str,
                                        r_t_paths: list,
                                        sigma: float,
                                        strike_price: float,
                                        time_t: float,
                                        time_0_f_curve: list,
                                        time_0_P_curve: list,
                                        units_basis_points: bool,
                                        x_t_paths: list,
                                        zero_coupon_bond: list,
                                        verbose: bool = False
                                       ) -> float:
    """
    Info: This function computes the exact price of a zero-coupon bond call or 
        put option in the one-factor Hull-White (1F-HW) short rate model.
    
    Input:
        a_param: a float specifying the constant mean reversion rate parameter 
            in the 1F-HW short rate model.
        
        maturity_bond: a float specifying the maturity in years of the 
            underlying zero-coupon bond.
        
        maturity_option: a float specifying the maturity in years of the option 
            on the zero-coupon bond.
            
        n_annual_trading_days: an int specifying the number of trading days 
            per year for the time axis discretization.
            
        option_type: a str specifying the type of the zero-coupon bond option: 
            can be 'call' or 'put'.
        
        r_t_paths: a 2D ndarray containing the simulated short rate paths along 
            a number of columns corresponding to the number of paths and a 
            number of rows being a discrete short rate time series of length 
            equal to the total number of trading days.
        
        sigma: a float specifying the constant volatility factor in the 
            1F-HW short rate model.
        
        strike_price: a float specifying the strike price of the zero-coupon 
            bond option.
        
        time_t: a float specifying the time of evaluation in years.
        
        time_0_f_curve: an ndarray containing the time values of the market 
            time-zero instantaneous forward curve.

        time_0_P_curve: an ndarray containing the time values of the market 
            time-zero zero-bond curve.
            
        units_basis_points: a bool which if True causes the output value to be 
            given in basis points of the notional.
            
        x_t_paths: a 2D ndarray containing the zero-mean process paths along 
            a number of columns corresponding to the number of paths and a 
            number of rows being a discrete short rate time series of length 
            equal to the total number of trading days.
        
        zero_coupon_bond: the array containing previously priced 
            zero-coupon bonds.
            
        verbose: a bool which if False blocks the function prints.
        
    Output:
        option_prices: an ndarray containing the pathwise prices of the 
            zero-coupon bond option.
    """
    # print(100*'*')
    # print(f'Zero-coupon bond {option_type} option valuation function initialized.\n')
    
    ## Check whether evaluation time t does not occur after option maturity
    if time_t > maturity_option:
        raise ValueError('evaluation time t may not occur after option maturity.')
    else:
        t_idx = int(time_t*n_annual_trading_days)
    
    ## Check whether option type was specified correctly and assign the delta 
    # parameter of the payoff function
    if option_type.lower() == 'call':
        delta_payoff = 1
    elif option_type.lower() == 'put':
        delta_payoff = -1
    else:
        raise ValueError('zero-coupon bond option type not detected.' 
                         + ' Enter as "put" or "call"')
        
    if r_t_paths is None and x_t_paths is None:
        raise ValueError('no short rate paths or zero-mean process paths' 
                          + ' were passed.')
        
    ## Price the zero-coupon bonds P(t, T) and P(t, S) if the latter was not 
    ## provided
    if x_t_paths is not None:
        P_t_T = price_zero_coupon_bond(a_param, n_annual_trading_days, 
                                       None, sigma, time_t, maturity_option, 
                                       time_0_f_curve, time_0_P_curve, 
                                       x_t_paths[t_idx])
        
        if zero_coupon_bond is not None:
            P_t_S = zero_coupon_bond
        else:
            P_t_S = price_zero_coupon_bond(a_param, n_annual_trading_days, 
                                           None, sigma, time_t, maturity_bond, 
                                           time_0_f_curve, time_0_P_curve, 
                                           x_t_paths[t_idx])
            
    elif r_t_paths is not None and x_t_paths is None:
        P_t_T = price_zero_coupon_bond(a_param, n_annual_trading_days, 
                                       r_t_paths[t_idx], sigma, time_t, 
                                       maturity_option, time_0_f_curve, 
                                       time_0_P_curve, None)
        
        if zero_coupon_bond is not None:
            P_t_S = zero_coupon_bond
        else:
            P_t_S = price_zero_coupon_bond(a_param, n_annual_trading_days, 
                                           r_t_paths[t_idx], sigma, time_t, 
                                           maturity_bond, time_0_f_curve, 
                                           time_0_P_curve, None)
        
    ## Evaluate the sigma_p and h parameter values as defined below Equation 
    # (3.40) on p. 76 of ref. [1]
    B_T_S = B_func(a_param, maturity_option, maturity_bond)
    sigma_p = sigma*np.sqrt((1 - np.exp(-2*a_param*(maturity_option - time_t)))
                            /(2*a_param))*B_T_S
    h = 1/sigma_p*np.log(P_t_S/(P_t_T*strike_price)) + sigma_p/2
    
    ## Evaluate the option price using Equations (3.40) and (3.41) on p. 76 of 
    # ref. [1]
    option_prices = delta_payoff*(P_t_S*st.norm.cdf(delta_payoff*h) 
                                  - strike_price*P_t_T
                                  *st.norm.cdf(delta_payoff*(h - sigma_p)))
    
    if verbose:
        if len(option_prices) == 1:
            print(f'Zero-coupon bond {option_type.lower()} option price at time'
                  + f' t={time_t}: {option_prices}')
        else:
            print(f'Mean zero-coupon bond {option_type.lower()} option price at time'
                  + f' t={time_t}: {np.mean(option_prices)}')
    
    return option_prices*10**4 if units_basis_points else option_prices

def price_zero_coupon_bond_option_MC(a_param: float,
                                     maturity_bond: float,
                                     maturity_option: float,
                                     n_annual_trading_days: int,
                                     option_type: str,
                                     r_t_paths: list,
                                     sigma: float,
                                     strike_price: float,
                                     time_t: float,
                                     time_0_f_curve: list,
                                     time_0_P_curve: list,
                                     units_basis_points: bool,
                                     x_t_paths: list,
                                     zero_coupon_bond: list,
                                     verbose: bool = False
                                    ) -> float:
    """
    Info: This function computes the Monte Carlo price of a zero-coupon bond 
        call or put option in the one-factor Hull-White (1F-HW) short rate 
        model.
    
    Input:
        a_param: a float specifying the constant mean reversion rate parameter 
            in the 1F-HW short rate model.
        
        maturity_bond: a float specifying the maturity in years of the 
            underlying zero-coupon bond.
        
        maturity_option: a float specifying the maturity in years of the option 
            on the zero-coupon bond.
            
        n_annual_trading_days: an int specifying the number of trading days 
            per year for the time axis discretization.
            
        option_type: a str specifying the type of the zero-coupon bond option: 
            can be 'call' or 'put'.
        
        r_t_paths: a 2D ndarray containing the simulated short rate paths along 
            a number of columns corresponding to the number of paths and a 
            number of rows being a discrete short rate time series of length 
            equal to the total number of trading days.
        
        sigma: a float specifying the constant volatility factor in the 
            1F-HW short rate model.
        
        strike_price: a float specifying the strike price of the zero-coupon 
            bond option.
        
        time_t: a float specifying the time of evaluation in years.
        
        time_0_f_curve: an ndarray containing the time values of the market 
            time-zero instantaneous forward curve.

        time_0_P_curve: an ndarray containing the time values of the market 
            time-zero zero-bond curve.
            
        units_basis_points: a bool which if True causes the output value to be 
            given in basis points of the notional.
            
        x_t_paths: a 2D ndarray containing the zero-mean process paths along 
            a number of columns corresponding to the number of paths and a 
            number of rows being a discrete short rate time series of length 
            equal to the total number of trading days.
        
        zero_coupon_bond: the array containing previously priced 
            zero-coupon bonds.
            
        verbose: a bool which if False blocks the function prints.
            
    Output:
        option_prices: 
            The pathwise prices of the zero-coupon bond option.
    """
    # print(100*'*')
    # print(f'Zero-coupon bond {option_type} option valuation function initialized.\n')
    
    ## Check whether evaluation time t does not occur after option maturity
    if time_t > maturity_option:
        raise ValueError('evaluation time t may not occur after option maturity.')
    
    ## Check whether option type was specified correctly and assign the delta 
    # parameter of the payoff function
    if option_type.lower() == 'call':
        delta_payoff = 1
    elif option_type.lower() == 'put':
        delta_payoff = -1
    else:
        raise ValueError('zero-coupon bond option type not detected.' 
                         + ' Enter as "put" or "call"')
        
    if r_t_paths is None and x_t_paths is None:
        raise ValueError('no short rate paths or zero-mean process paths' 
                          + ' were passed.')
        
    ## Price the underlying zero-coupon bond P(t, S) if not provided
    if zero_coupon_bond is not None:
        P_T_S = zero_coupon_bond
    else:
        T_idx = int(maturity_option*n_annual_trading_days)
        P_T_S = price_zero_coupon_bond(a_param, n_annual_trading_days, 
                                       r_t_paths[T_idx], sigma, 
                                       maturity_option, maturity_bond, 
                                       time_0_f_curve, time_0_P_curve, 
                                       x_t_paths[T_idx])
    
    ## Evaluate the discount factors and option prices
    discount_factors = eval_discount_factors(n_annual_trading_days, 
                                             r_t_paths, time_t, 
                                             maturity_option)
    option_prices = discount_factors*np.maximum(delta_payoff
                                                *(P_T_S - strike_price), 0.)
    
    if verbose:
        if len(option_prices) == 1:
            print(f'Zero-coupon bond {option_type.lower()} option price at time'
                  + f' t={time_t}: {option_prices}')
        else:
            print(f'Mean zero-coupon bond {option_type.lower()} option price at time'
                  + f' t={time_t}: {np.mean(option_prices)}')
    
    return option_prices*10**4 if units_basis_points else option_prices