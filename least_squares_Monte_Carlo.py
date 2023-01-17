# -*- coding: utf-8 -*-

# References:
#     [1] Longstaff, F. A., & Schwartz, E. S. (2001). Valuing American Options 
#     by Simulation: A Simple Least-Squares Approach. Review of Financial 
#     Studies, 14 (1). Retrieved from 
#     https://econpapers.repec.org/RePEc:oup:rfinst:v:14:y:2001:i:1:p:113-47
#     [2] Brigo, D., & Mercurio, F. (2006). Interest Rate Models - Theory and 
#         Practice: With Smile, Inflation, and Credit. Springer, Berlin, 
#         Heidelberg. doi: 10.1007/978-3-540-34604-3
#     [3] Glasserman, P. (2003). Monte Carlo Methods in Financial Engineering. 
#         Springer. doi: 10.1007/978-0-387-21617-1


# Imports
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.stats as si

# Local imports
from Black_Scholes_model import (generate_stock_paths_exact_cumsum, 
                                 price_European_stock_option_exact)
from bonds_and_bond_options import (construct_zero_coupon_bonds_curve,
                                    price_zero_coupon_bond)
from interest_rate_functions import (eval_annuity_terms,
                                     eval_simply_comp_forward_rate,
                                     eval_swap_rate)
from one_factor_Hull_White_model import (eval_discount_factors)
from swaps import (eval_moneyness_adjusted_fixed_rate, 
                   price_forward_start_swap)

def price_Bermudan_stock_option_LSM(degree: int,
                                    experiment_dir: str,
                                    n_annual_exercise_dates: int,
                                    n_paths: int,
                                    option_type: str,
                                    plot: bool,
                                    plot_regression: bool,
                                    r: float,
                                    r_t_paths: list,
                                    regression_series: str,
                                    S_0: float,
                                    sigma: float,
                                    sim_time: float,
                                    strike_price: float,
                                    verbose: bool = False
                                   ) -> list:
    """
    Info: 
        This function computes the value of a vanilla Bermudan stock call or 
        put option using the least-squares Monte Carlo (LSM) method as 
        introduced by Longstaff & Schwartz in ref. [1].
        
    Input: 
        degree: an int specifying the degree of the polynomial used in the 
            regression.
        
        experiment_dir: the directory to which the results are saved.
        
        input_stock_paths: an optional input array of stock paths to be used 
            as the evolution of the underlying stock of the option.
            
        n_annual_exercise_dates: an int specifying the number of trading days 
            per year for the time axis discretization which coincide with the 
            exercise dates.
        
        n_paths: an int specifying the number of stock paths to be simulated 
            in case no input stock paths were passed.
            
        option_type: a str specifying the option type as 'call' or 'put'.
        
        plot: a bool which if True plots the underlying stock paths which are 
            then saved to the local folder.
            
        plot_regression: a bool which if True plots the regression at a select 
            number of intermediate time steps.
            
        r: a float specifying the constant, continuously-compounded interest 
            rate.
        
        regression_series: a str specifying the polynomial type used in the 
            regression which can be either 'LaGuerre' or 'power'.
            
        S_0: a float specifying the initial stock value.
        
        sigma: a float specifying the constant volatility factor of the stock 
            paths.
        
        sim_time: a float specifying the simulation time in years.
        
        strike_price: a float specifying the strike price of the option.
        
        verbose: a bool which if False blocks the function prints.
        
    Output:
        mean_value: a float specifying the mean LSM value of the option.
        
        standard_error: a float specifying standard error corresponding to the 
            computed mean LSM option value.
    """
    
    # Simulate stock paths if no stock paths were passed in an input array
    if r_t_paths is None:
        print('\nGenerating stock paths for optimal stopping time determination.')
        stock_path_matrix = generate_stock_paths_exact_cumsum(
                                                n_annual_exercise_dates, 'Q', 
                                                None, n_paths, False, r, S_0, 
                                                sigma, sim_time)
    else:
        print('Taking input stock paths for optimal stopping time determination.')
        stock_path_matrix = r_t_paths
        n_paths = stock_path_matrix.shape[1]
    
    # Determine the option pay-off type
    if option_type.lower() == 'call':
        payoff_type = 1
    elif option_type.lower() == 'put':
        payoff_type = -1
    else:
        raise ValueError('option_type must be "call" or "put"')
    
    # Evaluate the total number of trading days, the spacing of the time axis 
    # delta_t, and the constant discount factor
    n_trading_days = stock_path_matrix.shape[0] - 1
    delta_t = sim_time/n_trading_days
    discount_factor = np.exp(-r*delta_t)
    
    ## Evaluate the option payoff and cashflow matrices. The payoff matrix will 
    ## remain unchanged throughout the algorithm while the cashflow matrix will 
    ## be altered as the option is either exercised or held at various time 
    ## points
    payoff_matrix = np.maximum(payoff_type*(stock_path_matrix - strike_price), 
                               0)
    cashflow_matrix = np.copy(payoff_matrix[1:])
    
    # Iterate backward over the stock paths and at each time point before 
    # maturity whether the option should be exercised or held for each path
    for t_idx in range(cashflow_matrix.shape[0]-1)[::-1]:
        # Select the current cashflows
        current_cashflows = cashflow_matrix[t_idx]
        
        # Determine the indices of the stock paths that are currently
        # in-the-money
        itm_path_indices = np.nonzero(current_cashflows)[0]
        
        # Select the current stock values. Note that time t in the cashflow 
        # matrix corresponds to t+1 in the stock path and payoff matrices
        current_stock_values = stock_path_matrix[t_idx+1,itm_path_indices]
        
        # Initialize array for storing the discounted future cashflows for each 
        # currently in-the-money stock path
        discounted_future_cashflows = np.zeros(len(itm_path_indices))
        
        # For each currently in-the-money stock paths, take all non-zero 
        # future cashflows and determine which one has the largest value 
        # after discounting back to the current time point
        for count, p_idx in enumerate(itm_path_indices):
            future_cashflows = cashflow_matrix[t_idx+1:,p_idx]
            discount_factors_vector = np.power(discount_factor, 
                                               np.arange(len(future_cashflows))+1)
            max_discounted_future_cashflow_idx = np.argmax(discount_factors_vector
                                                           *future_cashflows)
            discounted_future_cashflows[count] = (discount_factors_vector[max_discounted_future_cashflow_idx]
                                                  *future_cashflows[max_discounted_future_cashflow_idx])
            
        # If at least one path is in-the-money:
        if not itm_path_indices.size == 0:
            if regression_series.lower() == 'laguerre':
                # Determine the conditional expectation function by regressing 
                # the discounted future cashflows on the current stock values
                cond_exp_func = np.polynomial.laguerre.lagfit(
                                                current_stock_values, 
                                                discounted_future_cashflows, 
                                                degree)
                
                # Evaluate the continuation values
                continuation_values = np.polynomial.laguerre.lagval(
                                                        current_stock_values, 
                                                        cond_exp_func)
            elif regression_series.lower() == 'power':
                # Determine the conditional expectation function by regressing 
                # the discounted future cashflows on the current stock values
                cond_exp_func = np.polynomial.polynomial.polyfit(
                                                current_stock_values, 
                                                discounted_future_cashflows, 
                                                degree)
                # Evaluate the continuation values
                continuation_values = np.polynomial.polynomial.polyval(
                                                        current_stock_values,
                                                        cond_exp_func)
            # For each stock path, determine whether it will be exercised at 
            # the current time or held until the next non-zero cashflow
            cashflow_matrix[t_idx,itm_path_indices] = np.where(continuation_values 
                                                               > current_cashflows[itm_path_indices], 
                                                               0, 
                                                               current_cashflows[itm_path_indices])
            
            # Determine the indices of the paths that were exercised in order 
            # to set all their future cashflows to zero
            exercised_itm_indices = np.nonzero(cashflow_matrix[t_idx])[0]
            cashflow_matrix[t_idx+1:,exercised_itm_indices] = 0
        
            # Plot the regression at intermediate time points if selected by
            # user
            if plot_regression == True:
                # Plot least squares fit every so often
                if t_idx%20 == 0:
                    # Plot regression:
                    plt.scatter(current_stock_values, 
                                discounted_future_cashflows, 
                                label='Discounted Future Cashflows')
                    plt.scatter(current_stock_values, 
                                continuation_values, color='black',
                             label='Continuation Values', s=1)
                    plt.xlabel(f'Current Stock Value $S_t$ for $t = {t_idx+1}$')
                    plt.ylabel('Value')
                    plt.title('Least Squares Fit of' 
                              + f' {regression_series.capitalize()} Series')
                    plt.legend(loc='best')
                    plt.show()
                    plt.close()
                    
                    
    # Initialize array for storing all the optimal exercise strategy
    live_paths_array = np.zeros(n_paths, dtype=int)
    
    # Construct vector of discount factors to find the optimal pathwise 
    # stopping times 
    discount_factors_vector = np.power(discount_factor, 
                                       np.arange(1, n_trading_days+1)+1)
    
    # Loop over all paths and determine the non-zero cashflows then compute the 
    # optimal stopping times
    for p_idx in range(n_paths):
        stopping_times = np.nonzero(cashflow_matrix[:,p_idx])[0]
        if stopping_times.size > 1:
            # If there are multiple non-zero future cashflows, select the one 
            # that has the largest value after discounting back to time 1 and 
            # add it to the exercise strategy array
            optimal_stopping_time_idx = np.argmax(discount_factors_vector
                                                  *cashflow_matrix[:,p_idx])
            live_paths_array[p_idx] = optimal_stopping_time_idx + 1
        elif stopping_times.size == 1:
            # If there is only one non-zero future cashflow, select it and add 
            # it to the exercise strategy array
            live_paths_array[p_idx] = stopping_times[0] + 1
        else:
            # If there are no non-zero future cashflows, set the stopping time 
            # to -1 to let it expire
            live_paths_array[p_idx] = -1
    
    # Reuse the originally simulated stock paths for valuation of the option 
    # using the optimal stopping times
    option_price_matrix = np.zeros(n_paths)
    
    # Loop over all orginally simulated stock paths and evaluate the option 
    # value for each optimal stopping time
    for p_idx in range(n_paths):
        # Let the option expire if the assigned optimal stopping time was set 
        # to -1
        if live_paths_array[p_idx] == -1:
            option_price_matrix[p_idx] = 0
        # Evaluate the option payoff for each non-negative stopping time and 
        # discount back to time zero to find the pathwise option value
        else:
            option_price_matrix[p_idx] = (np.exp(-r
                                                *live_paths_array[p_idx]*delta_t)
                                          *payoff_matrix[live_paths_array[p_idx],p_idx])
    
    # Compute the mean LSM value of the option as well as the corresponding 
    # standard error
    mean_value = np.mean(option_price_matrix, dtype=np.float64)
    standard_error = si.sem(option_price_matrix)
    
    # Compute the value of the analogous European option
    Euro_val = price_European_stock_option_exact(sim_time, option_type, r, S_0, 
                                                 strike_price, sigma, 0)
    
    if verbose:
        # Print the LSM value, standard error, and early exercise value
        print(f'Option price: {mean_value}. Standard error: {standard_error}. '
              + f'Early exercise value compared to European option: {mean_value - Euro_val}.\n')

    return mean_value, standard_error

def price_Bermudan_swaption_LSM_Q(a_param: float, 
                                  degree: int,
                                  experiment_dir: str,
                                  fixed_rate: float,
                                  moneyness: float,
                                  n_annual_trading_days: int,
                                  notional: float,
                                  payoff_var: str,
                                  plot_regression: bool,
                                  plot_timeline: bool,
                                  regression_series: str,
                                  r_t_paths: list,
                                  sigma: float,
                                  swaption_type: str,
                                  tenor_structure: list,
                                  time_t: float,
                                  time_0_df0t_curve: list,
                                  time_0_f_curve: list,
                                  time_0_P_curve: list, 
                                  units_basis_points: bool,
                                  x_t_paths: list,
                                  verbose: bool = False
                                 ) -> list:
    """
    Info: This function computes the Least Squares Monte Carlo (LSM) price of 
        a Bermudan swaption using the One-Factor Hull & White (1F-HW) short 
        rate model under the risk-neutral measure Q.
        
    Input: 
        a_param: a float specifying the constant mean reversion rate parameter 
            in the 1F-HW short rate model.
            
        degree: an int specifying the degree of the polynomial used in the 
            regression.
            
        experiment_dir: the directory to which the results are saved.
        
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
            
        plot_regression: a bool which if True plots the regression at a select 
            number of intermediate time steps.
            
        plot_timeline: a bool specifying whether or not the (underlying) swap 
            timeline is plotted and saved to the local folder.
            
        regression_series: a str specifying the polynomial type used in the 
            regression which can be either 'LaGuerre' or 'power'.
            
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
        
        time_0_dfdt_curve: an ndarray containing the time derivative values of 
            the time-zero instantaneous forward curve.
            
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
        swaption_price_vector: a 1D ndarray containing the pathwise LSM prices 
            of the Bermudan swaption.
            
        live_paths_array: a 2D ndarray containing the live status of the 
            Monte Carlo paths on the monitor dates.
            
        pathwise_stopping_times: 2 2D ndarray containing the pathwise stopping 
            times of the Bermudan swaption.
    """
    # Check whether Data directory located in current directory and if not, 
    # create Data directory
    data_dir = os.path.join(experiment_dir, 'Data\\')

    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    
    # Check whether swaption type was specified correctly and assign the delta 
    # parameter of the payoff function
    if (swaption_type.lower() != 'payer' 
        and swaption_type.lower() != 'receiver'):
        raise ValueError('forward-start swaption type not detected. Enter as ' 
                         + '"payer" or "receiver"')
        
    # Check whether short rate paths and/or zero-mean path were passed
    if r_t_paths is None:
        raise ValueError('no short rate paths were passed.')
        
    # If specified, compute the moneyness-adjusted fixed rate
    if moneyness is not None:
        fixed_rate = eval_moneyness_adjusted_fixed_rate(moneyness, 
                                                        n_annual_trading_days, 
                                                        swaption_type, 
                                                        tenor_structure, 
                                                        time_0_P_curve)
    
    # Check whether the payoff function type was specified correctly
    if payoff_var.lower() != 'swap' and payoff_var.lower() != 'forward':
        raise ValueError('no swaption payoff variable was specified.' 
                         + ' Enter as "swap" or "forward".')
    
    # Determine the monitor dates that are relevant at the pricing time and 
    # adjust the tenor structure accordingly
    first_monitor_date_idx = np.searchsorted(tenor_structure, time_t)
    print(f'time_t: {time_t}')
    print(f'first_monitor_date_idx: {first_monitor_date_idx}')
    tenor_structure = tenor_structure[first_monitor_date_idx:]
    
    # Evaluate the number of paths and the tenor
    n_paths = r_t_paths.shape[1]
    tenor = len(tenor_structure)
    
    # Initialize array for storing the pathwise payoffs
    payoff_matrix = np.zeros((tenor-1, n_paths))
    
    ## Evaluate the option payoff and cashflow matrices. The payoff matrix will 
    ## remain unchanged throughout the algorithm while the cashflow matrix will 
    ## be altered as the option is either exercised or held at various time 
    ## points
    
    # Iterate over the monitor dates and evaluate the corresponding payoffs
    for count, time_T_i in enumerate(tenor_structure[:-1]):
        # Evaluate pathwise zero-coupon bond prices from current monitor date 
        # T_current to the final payment date T_M for evaluating the 
        # swaption payoff
        if verbose:
            print(f'Evaluating pay-off matrix at T = {time_T_i}')
        
        payoff_matrix[count] = np.maximum(price_forward_start_swap(a_param, 
                                        fixed_rate, n_annual_trading_days, 
                                        notional, payoff_var, plot_timeline, 
                                        r_t_paths, sigma, swaption_type, 
                                        tenor_structure[count:], time_T_i, 
                                        time_0_f_curve, time_0_P_curve, False, 
                                        x_t_paths, verbose), 0.)
        
    ## Apply the LSM algorithm in order to find the optimal stopping times
    cashflow_matrix = np.copy(payoff_matrix)
    
    # Iterate backward over the monitor dates excluding the last one
    for count, time_T_i in enumerate(tenor_structure[-3::-1]):
        # Determine the indices corresponding to the current monitor date in 
        # the tenor_structure vector and the payment date in the short rate 
        # matrix
        T_current_idx = int(time_T_i*n_annual_trading_days)
        tenor_idx = tenor - 3 - count
        
        # Select the cashflows corresponding to the current monitor dates
        current_cashflows = cashflow_matrix[tenor_idx]
        
        # Determine the indices of the short rate paths for which the swaption 
        # is in-the-money at the current monitor date T_current and store the 
        # scorresponding short rate values in array their array
        itm_path_idxs = np.nonzero(current_cashflows)[0]
        current_x_t = x_t_paths[T_current_idx,itm_path_idxs]
        
        # Initialize array for storing the discounted future cashflows for each 
        # currently in-the-money short rate path
        discounted_future_cashflows = np.zeros(len(itm_path_idxs))
        
        # For each currently in-the-money path, take all non-zero future 
        # cashflows and determine which one has the largest value after 
        # discounting back to the current monitor date
        for count2, p_idx in enumerate(itm_path_idxs):
            # Determine the number of future cashflows not directly following 
            # the current monitor date and initialize array for storing them
            n_future_cashflows = cashflow_matrix.shape[0] - 1 - tenor_idx
            future_cashflows = cashflow_matrix[tenor_idx+1:,p_idx]
            
            # Initialize and compute the vector storing the discount factors 
            # for discounting from each of the future cashflow dates back to 
            # the current monitor date
            discount_factors_vector = np.zeros(n_future_cashflows)
            
            for count3, T_idx in enumerate(range(n_future_cashflows)):
                discount_factors_vector[count3] = eval_discount_factors(
                                                    n_annual_trading_days, 
                                                    r_t_paths[:,p_idx], 
                                                    time_T_i, 
                                                    tenor_structure[tenor_idx+T_idx+1])
            
            # Determine the index of the maximum discounted future cashflow and 
            # store the corresponding discounted cashflow in the discounted 
            # future cashflows array
            max_idx = np.argmax(discount_factors_vector*future_cashflows)
            discounted_future_cashflows[count2] = (discount_factors_vector[max_idx]
                                                    *future_cashflows[max_idx])
            
        # # Evaluate the discounted future cashflows
        # Proceed if at least one path is in-the-money
        if not itm_path_idxs.size == 0:
            if regression_series.lower() == 'laguerre':
                # Determine the conditional expectation function by regressing 
                # the discounted future cashflows on the current short rate 
                # values using Laguerre polynomials
                cond_exp_func = np.polynomial.laguerre.lagfit(
                                                current_x_t, 
                                                discounted_future_cashflows, 
                                                degree)
                
                # Evaluate the continuation values
                continuation_values = np.polynomial.laguerre.lagval(
                                                        current_x_t, 
                                                        cond_exp_func)
            elif regression_series.lower() == 'power':
                # Determine the conditional expectation function by regressing 
                # the discounted future cashflows on the current short rate 
                # values using power series
                cond_exp_func = np.polynomial.polynomial.polyfit(
                                                current_x_t, 
                                                discounted_future_cashflows, 
                                                degree)
                # Evaluate the continuation values
                continuation_values = np.polynomial.polynomial.polyval(
                                                        current_x_t, 
                                                        cond_exp_func)
            else:
                raise ValueError('regression series not recognized.' 
                                 + ' Enter as "Laguerre" or "power".')
            
            # For each in-the-money short rate path, determine whether the 
            # swaption will be exercised on the current monitor date or held 
            # until the monitor date that corresponds to the next non-zero 
            # cashflow
            cashflow_matrix[tenor_idx,
                            itm_path_idxs] = np.where(current_cashflows[itm_path_idxs] 
                                                      > continuation_values, 
                                                      current_cashflows[itm_path_idxs], 
                                                      0)
            
            # Determine the indices of the paths that were exercised in order 
            # to set all their future cashflows to zero
            exercised_itm_indices = np.nonzero(cashflow_matrix[tenor_idx])[0]
            cashflow_matrix[tenor_idx+1:,exercised_itm_indices] = 0
        
            # Plot the regression for each monitor date if selected by user
            if plot_regression == True:
                if tenor_idx%1 == 0:
                    # plt.scatter(current_x_t, 
                    #             discounted_future_cashflows,
                    #             label='Discounted Future Cashflows')
                    plt.scatter(current_x_t, 
                                cashflow_matrix[tenor_idx,itm_path_idxs],
                                label='Current Cashflows', marker='x')
                    plt.scatter(current_x_t, 
                                continuation_values, color='black',
                             label='Continuation Values', s=1)
                    plt.xlabel(r'Current $x_t$ for' 
                               + r' Monitor Date $T_{fix}$' +f' = {time_T_i}')
                    plt.ylabel('Value')
                    plt.title('Least Squares Fit of' 
                              + f' {regression_series.capitalize()} Series')
                    plt.legend(loc='best')
                    plt.show()
                    plt.close()
            
    # Initialize array for storing all the optimal exercise strategy
    # live_paths_array = np.zeros(n_paths, dtype=int)
    live_paths_array = np.ones((tenor-1, n_paths), dtype=int)
    pathwise_stopping_times = np.zeros_like(live_paths_array, dtype=int)
            
    # Construct and evaluate matrix of discount factors from payment dates back 
    # to option expiry to find the optimal exercise strategy
    expiry = tenor_structure[0]
    discount_factors_expiry_matrix = np.zeros((tenor-1, n_paths))
    
    for count, time_T_i in enumerate(tenor_structure[:-1]):
        discount_factors_expiry_matrix[count] = eval_discount_factors(
                                                        n_annual_trading_days, 
                                                        r_t_paths, 
                                                        expiry, time_T_i)
    
    # Loop over all paths and determine non-zero cashflows, then compute the 
    # optimal stopping times
    for p_idx in range(n_paths):
        stopping_times = np.nonzero(cashflow_matrix[:,p_idx])[0]
        if stopping_times.size > 1:
            # If there are multiple non-zero future cashflows, select the one 
            # that has the largest value after discounting back to swaption 
            # expiry and add it to the exercise strategy array
            optimal_stopping_time_idx = np.argmax(discount_factors_expiry_matrix[:,p_idx]
                                                  *cashflow_matrix[:,p_idx])
            live_paths_array[optimal_stopping_time_idx+1:,p_idx] = 0
            pathwise_stopping_times[optimal_stopping_time_idx,p_idx] = 1
            pathwise_stopping_times[optimal_stopping_time_idx+1:,p_idx] = 0
        elif stopping_times.size == 1:
            # If there is only one non-zero future cashflow, select it and add 
            # it to the exercise strategy array
            live_paths_array[stopping_times[0]+1:,p_idx] = 0
            pathwise_stopping_times[stopping_times[0],p_idx] = 1
            pathwise_stopping_times[stopping_times[0]+1:,p_idx] = 0
        else:
            # If there are no non-zero future cashflows, leave the live paths 
            # and stopping times arrays unchanged
            pass
        
        
    ## Price the swaption using the optimal stopping times
    # Reuse the simulated short rate paths for pricing of the swaption using 
    # the optimal stopping tim es
    swaption_price_vector = np.zeros(n_paths)
    
    # Loop over all previously simulated short rate paths and evaluate the 
    # swaption price for each optimal stopping time on the first monitor date
    for p_idx in range(n_paths):
        price =  (discount_factors_expiry_matrix[np.nonzero(live_paths_array[:,p_idx])[-1],p_idx]
                  *payoff_matrix[np.nonzero(live_paths_array[:,p_idx])[-1],p_idx])[-1]
        swaption_price_vector[p_idx] = price
            
    # Discount the swaption price back from expiry to time t
    discount_factors_t_vector = eval_discount_factors(n_annual_trading_days, 
                                                      r_t_paths, time_t, 
                                                      expiry)
    swaption_price_vector = discount_factors_t_vector*swaption_price_vector
            
    if verbose:
        # Print the LSM price and standard error
        mean_price = np.mean(swaption_price_vector, dtype=np.float64)
        standard_error = si.sem(swaption_price_vector)
        print(f'Bermudan {swaption_type} swaption price: {mean_price}.' 
              + f' Standard error: {standard_error}.')
    
    return ((swaption_price_vector*10**4/notional if units_basis_points 
            else swaption_price_vector), live_paths_array, 
            pathwise_stopping_times)