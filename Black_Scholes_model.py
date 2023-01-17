# -*- coding: utf-8 -*-

# Imports
from datetime import datetime
import numpy as np
import scipy.stats as si
from tqdm import trange

# Local imports
from plotting_functions import plot_time_series

# Valuation functions
def generate_stock_paths_Euler(measure: str,
                               mu: float,
                               n_annual_trading_days: int,
                               n_paths: int,
                               plot: bool,
                               r: float,
                               S_0: float,
                               sigma: float,
                               sim_time: float
                              ) -> list:
    """
    Info: 
        This function generates stock path time series under the real-world 
        measure P or the risk-neutral measure Q using geometric Brownian motion 
        (GBM) with Euler-Maruyama discretization. The stock paths are generated
        as antithetic pairs with respect to the Brownian increment.
    
    Input: 
        measure: a str specifying the probability measure under which the stock 
            paths evolve.
            
        mu: a float specifying the real-world drift.
        
        n_annual_trading_days: an int specifying the number of trading days 
            per year for the time axis discretization.
        
        n_paths: an int specifying the number of paths to be simulated.
        
        plot: a bool which if True plots the GBM paths which are then saved to 
            the local folder.
        
        r: a float specifying the riskless, continuously-compounded interest 
            rate.
        
        S_0: a float specifying the time-zero stock value.
        
        sigma: a float specifying the constant volatility parameter.
        
        sim_time: a float specifying the simulation time in years.
        
    Output:
        stock_paths: a 2D ndarray containing the GBM paths over time with the 
            time values given along the rows and the pathwise values given 
            along the columns.
    """
    if measure.lower() == 'real-world' or measure.lower() == 'p':
        drift = mu
    elif measure.lower() == 'risk-neutral' or measure.lower() == 'q':
        drift = r
    else:
        raise ValueError('measure must be "real-world"/"P" or "risk-neutral"/"Q"')
    
    # Evaluate total number of time points and equal-spacing time differential
    n_time_points = int(sim_time*n_annual_trading_days) + 1
    delta_t = sim_time/n_time_points # compute delta t
    
    # Initialize array for storing the sample paths
    stock_paths = np.zeros((n_time_points, n_paths))
    stock_paths[0,:] = S_0
    
    # Initialize individual sample path array
    stock_path = np.zeros(n_time_points)
    stock_path[0] = S_0
    stock_path_antithetic = np.copy(stock_path)
    
    print('\nGenerating antithetic pairs of stock paths...')
    
    # Loop over all paths
    for path in trange(-2, n_paths, 2):
        
        # Loop over all time points
        for idx in range(n_time_points - 1):
            # Sample from standard normal distribution
            Z = np.random.normal(0, 1)
            
            # Assign stock value at time next time point
            stock_path[idx+1] = (stock_path[idx] + drift*stock_path[idx]*delta_t 
                               + sigma*stock_path[idx]*np.sqrt(delta_t)*Z)
            stock_path_antithetic[idx+1] = (stock_path[idx] + drift*stock_path[idx]
                                          *delta_t + sigma*stock_path[idx]
                                          *np.sqrt(delta_t)*-Z)
            
        # Store stock path in matrix
        stock_paths[:,path] = stock_path
        stock_paths[:,path+1] = stock_path_antithetic
    
    if plot == True:
        # Plot the GBM paths
        plot_title = 'Stock Paths Using Euler Discretization of Geometric Brownian Motion'
        plot_time_series(None, None, False, plot_title, stock_paths, None, 
                         None, '$S_t$', None)
        plot_time_series(None, None, True, plot_title, stock_paths, None, None, 
                         '$S_t$', None)
    
    return stock_paths

def generate_stock_paths_exact(measure: str,
                               mu: float,
                               n_annual_trading_days: int,
                               n_paths: int,
                               plot: bool,
                               r: float,
                               S_0: float,
                               sigma: float,
                               sim_time: float
                              ) -> list:
    """
    Info: 
        This function generates stock path time series under the real-world 
        measure P or the risk-neutral measure Q using geometric Brownian motion 
        (GBM) with exact simulation and antithetic standard normal draws. The 
        stock paths are generated as antithetic pairs with respect to the 
        Brownian increment.
        
    Input: 
        measure: a str specifying the probability measure under which the stock 
            paths evolve.
            
        mu: a float specifying the real-world drift.
        
        n_annual_trading_days: an int specifying the number of trading days 
            per year for the time axis discretization.
        
        n_paths: an int specifying the number of paths to be simulated.
        
        plot: a bool which if True plots the GBM paths which are then saved to 
            the local folder.
        
        r: a float specifying the riskless, continuously-compounded interest 
            rate.
        
        S_0: a float specifying the time-zero stock value.
        
        sigma: a float specifying the constant volatility parameter.
        
        sim_time: a float specifying the simulation time in years.
        
    Output:
        stock_paths: a 2D ndarray containing the GBM paths over time with the 
            time values given along the rows and the pathwise values given 
            along the columns.
    """
    if measure.lower() == 'real-world' or measure.lower() == 'p':
        drift = mu
    elif measure.lower() == 'risk-neutral' or measure.lower() == 'q':
        drift = r
    else:
        raise ValueError('measure must be "real-world"/"P" or "risk-neutral"/"Q"')
    
    # Evaluate total number of time points and equal-spacing time differential
    n_time_points = int(sim_time*n_annual_trading_days) + 1
    delta_t = sim_time/n_time_points # compute delta t
    
    # Initialize array for storing the sample paths
    stock_paths = np.zeros((n_time_points, n_paths))
    stock_paths[0,:] = S_0
    
    # Initialize individual sample path array
    stock_path = np.zeros(n_time_points)
    stock_path[0] = S_0
    stock_path_antithetic = np.copy(stock_path)
    
    print('\nGenerating antithetic pairs of stock paths...')
    
    # Loop over all paths
    for path in trange(-2,n_paths,2):
        
        # Loop over all time points
        for idx in range(n_time_points - 1):
            # Sample from standard normal distribution
            Z = np.random.normal(0, 1)
            
            # Assign stock value at time next time point
            stock_path[idx+1] = stock_path[idx]*np.exp((drift - sigma**2/2)*delta_t 
                                                   + sigma*np.sqrt(delta_t)*Z)
            stock_path_antithetic[idx+1] = stock_path_antithetic[idx]*np.exp((drift - sigma**2/2)
                                                              *delta_t + sigma
                                                              *np.sqrt(delta_t)*-Z)
        
        # Store stock path in matrix
        stock_paths[:,path] = stock_path
        stock_paths[:,path+1] = stock_path_antithetic
    
    if plot == True:
        # Plot the GBM paths
        plot_title = 'Stock Paths Using Exact Simulation of Geometric Brownian Motion'
        plot_time_series(None, None, False, plot_title, stock_paths, None, 
                         None, '$S_t$', None)
        plot_time_series(None, None, True, plot_title, stock_paths, None, None, 
                         '$S_t$', None)
    
    return stock_paths

def generate_stock_paths_exact_cumsum(measure: str,
                                      mu: float,
                                      n_annual_trading_days: int,
                                      n_paths: int,
                                      plot: bool,
                                      r: float,
                                      S_0: float,
                                      sigma: float,
                                      sim_time: float
                                     ) -> list:
    """
    Info: 
        This function generates stock path time series under the real-world 
        measure P or the risk-neutral measure Q using geometric Brownian motion 
        (GBM) with exact simulation and antithetic standard normal draws. The 
        stock paths are generated as antithetic pairs with respect to the 
        Brownian increment.
        
    Input: 
        measure: a str specifying the probability measure under which the stock 
            paths evolve.
            
        mu: a float specifying the real-world drift.
        
        n_annual_trading_days: an int specifying the number of trading days 
            per year for the time axis discretization.
        
        n_paths: an int specifying the number of paths to be simulated.
        
        plot: a bool which if True plots the GBM paths which are then saved to 
            the local folder.
        
        r: a float specifying the riskless, continuously-compounded interest 
            rate.
        
        S_0: a float specifying the time-zero stock value.
        
        sigma: a float specifying the constant volatility parameter.
        
        sim_time: a float specifying the simulation time in years.
        
    Output:
        stock_paths: a 2D ndarray containing the GBM paths over time with the 
            time values given along the rows and the pathwise values given 
            along the columns.
    """
    # Evaluate total number of time points and equal-spacing time differential
    n_time_points = int(sim_time*n_annual_trading_days) + 1
    delta_t = sim_time/n_time_points
    
    
    print('\nGenerating antithetic pairs of stock paths...')
    starting_time = datetime.now()
    # Generate Brownian motions
    Zn = np.random.normal(0, 1, (n_time_points-1, int(n_paths/2)))
    
    stock_paths = np.zeros((n_time_points, n_paths))
    stock_paths[0] = S_0
    stock_paths[1:,::2] += S_0*np.exp(np.cumsum((r - sigma**2/2)*delta_t 
                                           + sigma*np.sqrt(delta_t)*Zn, axis=0))
    stock_paths[1:,1::2] += S_0*np.exp(np.cumsum((r - sigma**2/2)*delta_t 
                                           + sigma*np.sqrt(delta_t)*-Zn, axis=0))
    
    finish_time = datetime.now()
    print(f'{int(n_paths/2)} antithetic pairs of stock paths were generated in {(finish_time-starting_time).total_seconds()}.')
    print(starting_time, finish_time)
    
    if plot == True:
        # Plot the GBM paths
        plot_title = 'Stock Paths Using Exact Simulation of Geometric Brownian Motion'
        plot_time_series(None, None, True, plot_title, stock_paths, None, None, 
                         '$S_t$', None)
    
    return stock_paths
    
    

def price_European_stock_option_exact(maturity: float,
                                      option_type: str,
                                      r: float,
                                      S_t: float,
                                      strike_price: float,
                                      sigma: float,
                                      t: float
                                     ) -> list:
    """
    Info:
        This function computes the exact value of a European stock call or put 
        option at a specified time 0 <= t <= maturity using the Black-Scholes 
        formula.
    
    Input:
        maturity: a float specifying the option maturity in years.
        
        r: a float specifying the riskless, continuously-compounded interest 
        rate.
        
        S_t: a float specifying the stock value at time t.
        
        strike_price: a float specifying the strike of the option.
        
        sigma: a float specifying the volatility parameter.
        
        t: a float specifying the time of valuation in years.
    
    Output:
        price_t: an ndarray containing the pathwise stock option values.
    """
    # Check evaluation time
    if t > maturity:
        raise ValueError('t must be on the interval [0, maturity]')
    
    # Compute time-to-expiration tau and parameters d_1 and d_2
    tau = maturity - t
    d_1 = (np.log(S_t/strike_price) 
           + (r + sigma**2/2)*tau)/(sigma*np.sqrt(tau))
    d_2 = d_1 - sigma*np.sqrt(tau)
    
    # Evaluate the option type and option value using the Black-Scholes formula
    if option_type.lower() == 'call':
        N_d1 = si.norm.cdf(d_1, 0, 1)
        N_d2 = si.norm.cdf(d_2, 0, 1)
        price_t = S_t*N_d1 - strike_price*np.exp(-r*tau)*N_d2
    elif option_type.lower() == 'put':
        N_d1 = si.norm.cdf(-d_1, 0, 1)
        N_d2 = si.norm.cdf(-d_2, 0, 1)
        price_t = strike_price*np.exp(-r*tau)*N_d2 - S_t*N_d1
    else:
        raise ValueError('option_type must be "call" or "put"')
    
    return price_t

# Sensitivity functions
def delta_European_stock_option_exact(maturity: float,
                                      option_type: str,
                                      r: float,
                                      S_t: float,
                                      strike_price: float,
                                      sigma: float,
                                      t: float
                                     ) -> list:
    """
    Info:
        This function computes the exact value of the delta of a European call 
        or put option at a specified time 0 <= t <= maturity.
    
    Input:
        maturity: a float specifying the option maturity in years.
        
        r: a float specifying the riskless, continuously-compounded interest 
        rate.
        
        S_t: a float specifying the stock value at time t.
        
        strike_price: a float specifying the strike of the option.
        
        sigma: a float specifying the volatility parameter.
        
        t: a float specifying the time of valuation in years.
    
    Output:
        delta: a float specifying the partial derivative of the option value 
            with respect to the underlying stock value.
    """
    # Check evaluation time
    if t > maturity:
        raise ValueError('t must be on the interval [0, maturity]')
    
    # Compute time-to-expiration tau and parameter d_1
    tau = maturity - t
    d_1 = (np.log(S_t/strike_price) + (r + sigma**2/2)*tau)/(sigma*np.sqrt(tau))
    
    # Evaluate the option type and delta value
    if option_type.lower() == 'call':
        delta = si.norm.cdf(d_1, 0, 1)
    elif option_type.lower() == 'put':
        delta = -si.norm.cdf(-d_1, 0, 1)
    else:
        raise ValueError('option_type must be "call" or "put"')
    
    return delta

def gamma_European_stock_option_exact(maturity: float,
                                      option_type: str,
                                      r: float,
                                      S_t: float,
                                      strike_price: float,
                                      sigma: float,
                                      t: float
                                     ) -> list:
    """
    Info:
        This function computes the exact value of the gamma of a European call 
        or put option at a specified time 0 <= t <= maturity.
    
    Input:
        maturity: a float specifying the option maturity in years.
        
        r: a float specifying the riskless, continuously-compounded interest 
        rate.
        
        S_t: a float specifying the stock value at time t.
        
        strike_price: a float specifying the strike of the option.
        
        sigma: a float specifying the volatility parameter.
        
        t: a float specifying the time of valuation in years.
    
    Output:
        gamma: a float specifying the second partial derivative of the option 
            value with respect to the underlying stock value.
    """
    # Check evaluation time
    if t > maturity:
        raise ValueError('t must be on the interval [0, maturity]')
    
    # Check if correct option type was specified
    if option_type.lower() != 'call' and option_type.lower() != 'put':
        raise ValueError('option_type must be "call" or "put"')
        
    # Compute time-to-expiration tau and parameter d_1
    tau = maturity - t
    d_1 = (np.log(S_t/strike_price) + (r + sigma**2/2)*tau)/(sigma*np.sqrt(tau))
    
    # Evaluate the option type and delta value
    gamma = si.norm.pdf(d_1, 0, 1)/(S_t*sigma*np.sqrt(tau))
    
    return gamma

def vega_European_stock_option_exact(maturity: float,
                                     option_type: str,
                                     r: float,
                                     S_t: float,
                                     strike_price: float,
                                     sigma: float,
                                     t: float
                                    ) -> list:
    """
    Info:
        This function computes the exact value of the gamma of a European stock 
        call or put option at a specified time 0 <= t <= maturity.
    
    Input:
        maturity: a float specifying the option maturity in years.
        
        r: a float specifying the riskless, continuously-compounded interest 
        rate.
        
        S_t: a float specifying the stock value at time t.
        
        strike_price: a float specifying the strike of the option.
        
        sigma: a float specifying the volatility parameter.
        
        t: a float specifying the time of valuation in years.
    
    Output:
        gamma: a float specifying the second partial derivative of the option 
            value with respect to the underlying stock value.
    """
    # Check evaluation time
    if t > maturity:
        raise ValueError('t must be on the interval [0, maturity]')
    
    # Check if correct option type was specified
    if option_type.lower() != 'call' and option_type.lower() != 'put':
        raise ValueError('option_type must be "call" or "put"')
        
    # Compute time-to-expiration tau and parameter d_1
    tau = maturity - t
    d_1 = (np.log(S_t/strike_price) + (r + sigma**2/2)*tau)/(sigma*np.sqrt(tau))
    
    # Evaluate the option type and delta value
    vega = S_t*si.norm.pdf(d_1, 0, 1)*np.sqrt(tau)
    
    return vega

# Monte Carlo Greek functions
# Finite difference method
def delta_European_stock_option_bump_revalue(h: float,
                                             maturity: float,
                                             n_paths: int,
                                             option_type: str,
                                             r: float,
                                             S_0: float,
                                             strike_price: float,
                                             sigma: float,
                                             t: float
                                            ) -> list:
    """
    Info:
        This function computes the Monte Carlo value of the delta of a European 
        stock call or put option at a specified time 0 <= t <= maturity using 
        the central-difference estimator of the finite difference 
        bump-and-revalue method.
    
    Input:
        h: a float specifying the bump increment size.
        
        maturity: a float specifying the option maturity in years.
        
        r: a float specifying the riskless, continuously-compounded interest 
        rate.
        
        S_t: a float specifying the stock value at time t.
        
        strike_price: a float specifying the strike of the option.
        
        sigma: a float specifying the volatility parameter.
        
        t: a float specifying the time of valuation in years.
    
    Output:
        delta: an ndarray containing the pathwise partial derivatives of the 
            option value with respect to the underlying stock value.
    """
    # Check evaluation time
    if t > maturity:
        raise ValueError('t must be on the interval [0, maturity]')
    
    # Evaluate the payoff type
    if option_type.lower() == 'call':
        payoff_type = 1
    elif option_type.lower() == 'put':
        payoff_type = -1
    else:
        raise ValueError('option_type must be "call" or "put"')
    
    # Evaluate time-to-expiration tau
    tau = maturity - t
    
    if not h:
        # Set bump increment value to optimal order according to Glasserman
        # book Table 7.1 on p. 382
        h = n_paths**(-1/5)
    
    S_t_plus_h = np.zeros(n_paths)
    S_t_minus_h = np.zeros(n_paths)
    
    for idx in range(n_paths):
        # Use common random numbers to reduce variance
        Z = np.random.normal(0, 1)
        S_t_plus_h[idx] = (S_0 + h)*np.exp((r - sigma**2/2)*maturity 
                                                + sigma*np.sqrt(maturity)*Z)
        
        # Z = np.random.normal(0, 1)
        S_t_minus_h[idx] = (S_0 - h)*np.exp((r - sigma**2/2)*maturity 
                                                + sigma*np.sqrt(maturity)*Z)
    
    # Evaluate bumped payoffs
    payoff_plus_h = np.where(payoff_type*(S_t_plus_h - strike_price) > 0, 
                             payoff_type*(S_t_plus_h - strike_price), 0)
    payoff_minus_h = np.where(payoff_type*(S_t_minus_h - strike_price) > 0, 
                             payoff_type*(S_t_minus_h - strike_price), 0)
    
    # Evaluate delta
    delta = np.exp(-r*tau)*np.mean((payoff_plus_h - payoff_minus_h))/(2*h)
    
    return delta

def gamma_European_stock_option_bump_revalue(h: float,
                                             maturity: float,
                                             n_paths: int,
                                             option_type: str,
                                             r: float,
                                             S_0: float,
                                             strike_price: float,
                                             sigma: float,
                                             t: float
                                            ) -> list:
    """
    Info:
        This function computes the Monte Carlo value of the gamma of a European 
        stock call or put option at a specified time 0 <= t <= maturity using 
        the central-difference estimator of the finite difference 
        bump-and-revalue method.
    
    Input:
        h: a float specifying the bump increment size.
        
        maturity: a float specifying the option maturity in years.
        
        r: a float specifying the riskless, continuously-compounded interest 
        rate.
        
        S_t: a float specifying the stock value at time t.
        
        strike_price: a float specifying the strike of the option.
        
        sigma: a float specifying the volatility parameter.
        
        t: a float specifying the time of valuation in years.
    
    Output:
        gamma: an ndarray containing the pathwise second partial derivatives 
            of the option value with respect to the underlying stock value.
    """
    # Check evaluation time
    if t > maturity:
        raise ValueError('t must be on the interval [0, maturity]')
    
    # Evaluate the payoff type
    if option_type.lower() == 'call':
        payoff_type = 1
    elif option_type.lower() == 'put':
        payoff_type = -1
    else:
        raise ValueError('option_type must be "call" or "put"')
    
    # Evaluate time-to-expiration tau
    tau = maturity - t
    
    if not h:
        # Set bump increment value to optimal order according to Glasserman
        # book Table 7.1 on p. 382
        h = n_paths**(-1/5)
    
    S_t = np.zeros(n_paths)
    S_t_plus_h = np.zeros(n_paths)
    S_t_minus_h = np.zeros(n_paths)
    
    for idx in range(n_paths):
        # Use common random number to reduce variance
        Z = np.random.normal(0, 1)
        S_t = S_0*np.exp((r - sigma**2/2)*maturity 
                                                + sigma*np.sqrt(maturity)*Z)
        
        S_t_plus_h[idx] = (S_0 + h)*np.exp((r - sigma**2/2)*maturity 
                                                + sigma*np.sqrt(maturity)*Z)
        
        # Z = np.random.normal(0, 1)
        S_t_minus_h[idx] = (S_0 - h)*np.exp((r - sigma**2/2)*maturity 
                                                + sigma*np.sqrt(maturity)*Z)
    
    # Evaluate bumped payoffs
    payoff = np.where(payoff_type*(S_t - strike_price) > 0, 
                             payoff_type*(S_t - strike_price), 0)
    payoff_plus_h = np.where(payoff_type*(S_t_plus_h - strike_price) > 0, 
                             payoff_type*(S_t_plus_h - strike_price), 0)
    payoff_minus_h = np.where(payoff_type*(S_t_minus_h - strike_price) > 0, 
                             payoff_type*(S_t_minus_h - strike_price), 0)
    
    # Evaluate gamma
    gamma = np.exp(-r*tau)*np.mean(payoff_plus_h - 2*payoff 
                                    + payoff_minus_h)/(h**2)
    
    return gamma

def vega_European_stock_option_bump_revalue(h: float,
                                            maturity: float,
                                            n_paths: int,
                                            option_type: str,
                                            r: float,
                                            S_0: float,
                                            strike_price: float,
                                            sigma: float,
                                            t: float
                                           ) -> list:
    """
    Info:
        This function computes the Monte Carlo value of the vega of a European 
        stock call or put option at a specified time 0 <= t <= maturity using 
        the central-difference estimator of the finite difference 
        bump-and-revalue method.
    
    Input:
        h: a float specifying the bump increment size.
        
        maturity: a float specifying the option maturity in years.
        
        r: a float specifying the riskless, continuously-compounded interest 
        rate.
        
        S_t: a float specifying the stock value at time t.
        
        strike_price: a float specifying the strike of the option.
        
        sigma: a float specifying the volatility parameter.
        
        t: a float specifying the time of valuation in years.
    
    Output:
        vega: an ndarray containing the pathwise partial derivatives of the 
        option value with respect to the underlying deterministic volatility  
        value.
    """
    # Check evaluation time
    if t > maturity:
        raise ValueError('t must be on the interval [0, maturity]')
    
    # Evaluate the payoff type
    if option_type.lower() == 'call':
        payoff_type = 1
    elif option_type.lower() == 'put':
        payoff_type = -1
    else:
        raise ValueError('option_type must be "call" or "put"')
    
    # Evaluate time-to-expiration tau
    tau = maturity - t
    
    if not h:
        # Set bump increment value to optimal order according to Glasserman
        # book Table 7.1 on p. 382
        h = n_paths**(-1/5)
    
    r_plus_h = np.zeros(n_paths)
    r_minus_h = np.zeros(n_paths)
    
    for idx in range(n_paths):
        # Use common random numbers to reduce variance
        Z = np.random.normal(0, 1)
        r_plus_h[idx] = S_0*np.exp((r - (sigma + h)**2/2)*maturity 
                                                + (sigma + h)*np.sqrt(maturity)*Z)
        
        r_minus_h[idx] = S_0*np.exp((r - (sigma - h)**2/2)*maturity 
                                                + (sigma - h)*np.sqrt(maturity)*Z)
    
    # Evaluate bumped payoffs
    payoff_plus_h = np.where(payoff_type*(r_plus_h - strike_price) > 0, 
                             payoff_type*(r_plus_h - strike_price), 0)
    payoff_minus_h = np.where(payoff_type*(r_minus_h - strike_price) > 0, 
                             payoff_type*(r_minus_h - strike_price), 0)
    
    # Evaluate delta
    vega = np.exp(-r*tau)*np.mean((payoff_plus_h - payoff_minus_h))/(2*h)
    
    return vega

# Pathwise method
def delta_European_stock_option_pathwise(maturity: float,
                                         n_annual_trading_days: int,
                                         option_type: str,
                                         r: float,
                                         stock_paths: list,
                                         strike_price: float,
                                         sigma: float,
                                         t: float
                                        ) -> list:
    """
    Info:
        This function computes the Monte Carlo value of the delta of a European 
        stock call or put option at a specified time 0 <= t <= maturity using 
        the pathwise method.
    
    Input:
        maturity: a float specifying the option maturity in years.
        
        n_annual_trading_days: an int specifying the number of trading days 
            per year for the time axis discretization.
        
        r: a float specifying the riskless, continuously-compounded interest 
        rate.
        
        S_t: a float specifying the stock value at time t.
        
        strike_price: a float specifying the strike of the option.
        
        sigma: a float specifying the volatility parameter.
        
        t: a float specifying the time of valuation in years.
    
    Output:
        delta: an ndarray containing the pathwise partial derivatives of the 
            option value with respect to the underlying stock value.
    """
    # Check evaluation time
    if t > maturity:
        raise ValueError('t must be on the interval [0, maturity]')
    
    # Evaluate the payoff type
    if option_type.lower() == 'call':
        payoff_type = 1
    elif option_type.lower() == 'put':
        payoff_type = -1
    else:
        raise ValueError('option_type must be "call" or "put"')
    
    # Evaluate time-to-maturity tau and number of stock paths
    tau = maturity - t
    
    t_idx = int(t*n_annual_trading_days)
    indicator_vals = np.where(payoff_type*(stock_paths[-1] > strike_price), 1, 
                              0)
    pathwise_deltas = (np.exp(-r*tau)*indicator_vals*stock_paths[-1]
                       /stock_paths[t_idx])
        
    delta = np.mean(pathwise_deltas)
    
    return delta