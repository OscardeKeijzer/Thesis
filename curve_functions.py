# -*- coding: utf-8 -*-

# Imports
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
os.chdir(os.getcwd())
import pandas as pd
import scipy.stats as st
from tqdm import trange
import warnings
warnings.simplefilter('ignore', np.RankWarning)

# Local imports
from data_functions import write_Parquet_data
from plotting_functions import plot_time_series

def construct_time_0_zero_coupon_curve(experiment_dir: str,
                                       n_annual_trading_days: int,
                                       time_horizon: float,
                                       time_0_rate: float,
                                       plot_curve: bool = False,
                                       shape: str = 'flat',
                                       std_dev: float = 0.,
                                       verbose: bool = False
                                      ) -> list:
    """
    Info:
        This function computes the market time-zero zero rate curve R^M(0,t) 
        starting at an initial time-zero interest rate with a flat, decreasing, 
        or increasing curve structure.
        
    Input:
        experiment_dir: the directory to which the results are saved.
        
        n_annual_trading_days: the number of trading days per year, used as the 
            time axis discretization.
        
        plot_curve: a bool specifying whether or not the resulting curve is 
            plotted and saved to the local folder.
        
        shape: a str specifying what the shape of the time-zero zero rate curve 
            should be. 'flat' creates a flat curve, 'normal' creates an 
            increasing curve, and 'inverted' creates a decreasing curve.
            
        std_dev: the size of the standard deviation in the random number vector 
            that is generated in order to construct the zero rate curve.
            
        time_horizon: the time in years for which the market time-zero 
            zero rate curve will be constructed.
            
        time_0_rate: the time-zero interest rate on which the market time-zero 
            zero rate curve is based.
            
        verbose: a bool which if False blocks the function prints.
            
    Output:
        time_0_R_curve: an ndarray containing the time values of the time-zero 
            zero rate curve.
    """
    
    # Check whether Figures and Data directories located in current directory 
    # and if not, create them
    figures_dir = os.path.join(experiment_dir, 'Figures\\')
    data_dir = os.path.join(experiment_dir, 'Data\\')
    
    if not os.path.isdir(figures_dir):
        os.makedirs(figures_dir)
        
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    
    n_trading_days = int(time_horizon*n_annual_trading_days)
    
    # Construct time-zero zero rate curve using time-zero rate
    time_0_R_curve = np.random.normal(loc=time_0_rate, scale=std_dev, 
                                                size=n_trading_days 
                                                + int(n_annual_trading_days/2))
    
    if shape == 'flat':
        if verbose:
            print('\nFlat time-zero zero rate curve has been constructed.')
    elif shape == 'normal':
        # Transform flat time-zero zero rate curve to normal
        for i in range(len(time_0_R_curve)):
            time_0_R_curve[i] += np.sqrt(i)/(len(time_0_R_curve)*10)

        if verbose:
            print('\nNormal time-zero zero rate curve has been constructed.')
    elif shape == 'inverted':
        # Transform flat time-zero zero rate curve to inverted
        for i in range(len(time_0_R_curve)):
            time_0_R_curve[i] -= np.sqrt(i)/(len(time_0_R_curve)*10)
            
        if verbose:
            print('\nInverted time-zero zero rate curve has been constructed.')
    else:
        raise ValueError('time-zero zero rate curve shape not recognized.' 
                         + ' Enter as "flat", "normal", or "inverted".')
        
    # Save time-zero zero rate curve data to Parquet file with current date 
    # and time in filename
    if verbose:
        print('\nSaving data in Parquet format with brotli compression...')
    file_dir_and_name = str(data_dir 
                            + 'Time-Zero zero rate Curve'.replace(' ','_') 
                            + '-' + datetime.now().strftime('%Y-%m-%d_%H%M%S') 
                            + '.parquet')
    write_Parquet_data(experiment_dir, file_dir_and_name, time_0_R_curve)
    if verbose:
        print(f'\nTime-zero zero rate curve data saved to {file_dir_and_name}')
    
    # Plot time-zero zero rate curve and save image current date and time in 
    # filename
    if plot_curve == True:
        plot_title = 'Time-Zero Zero Rate Curve'
        x_label = 'Time $t$'
        y_label = '$R^M(0,t)$'
        
        if verbose:
            print('\nPlotting time-zero zero rate curve...')
        plot_time_series(experiment_dir, n_annual_trading_days, False, 
                         plot_title, time_0_R_curve, None, x_label, None, 
                         y_label, None)
    
    if verbose:
        print('Time-zero zero rate curve function terminated.\n\n')
    
    return time_0_R_curve

def bump_time_0_zero_coupon_curve(bump_time: float,
                                  experiment_dir: str,
                                  n_annual_trading_days: int,
                                  plot_curve: bool,
                                  time_0_R_curve: list,
                                  bump_size: float = .0001,
                                  verbose: bool = False
                                 ) -> list:
    """
    Info:
        This function bumps a previously constructed time-zero zero rate curve 
        for use in the bump-and-revalue method according to the ISDA-SIMM 
        specifications.
        
    Input:
        bump_size: the size of the bump for use in the resimulation of a bumped 
            set of short rates in the bump-and-revalue sensitivity estimation.
            
        bump_time: the time of the zero rate curve bump for use in the
            resimulation of a bumped set of short rates in the bump-and-revalue 
            sensitivity estimation method. Can be entered in years as type 
            float or as a string specifying one of the ISDA-SIMM tenors 
            {2w, 1m, 3m, 6m, 1Y, 2Y, 3Y, 5Y, 10Y, 15Y, 20Y, 30Y} formatted as 
            "ISDA-SIMM {tenor}".
            
        experiment_dir: the directory to which the results are saved.
        
        n_annual_trading_days: the number of trading days per year, used as the 
            time axis discretization.
        
        plot_curve: a bool specifying whether or not the resulting curve is 
            plotted and saved to the local folder.
        
        time_0_R_curve: an ndarray containing the time values of the time-zero 
            zero rate curve.
            
        verbose: a bool which if False blocks the function prints.
            
    Output:
        bumped_time_0_R_curve: an ndarray containing the time values of the 
            bumped time-zero zero rate curve.
    """
    
    # Check whether Figures and Data directories located in current directory 
    # and if not, create them
    figures_dir = os.path.join(experiment_dir, 'Figures\\')
    data_dir = os.path.join(experiment_dir, 'Data\\')
    
    if not os.path.isdir(figures_dir):
        os.makedirs(figures_dir)
        
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
        
    # Initialize the bumped time-zero zero rate curve array
    bumped_time_0_R_curve = np.copy(time_0_R_curve)
    
    # Instantiate dictionary containing ISDA-SIMM risk tenors
    ISDA_SIMM_tenors_dict = {'2w': 1/26, 
                             '1m': 1/12, 
                             '3m': 1/4, 
                             '6m': 1/2, 
                             '1y': 1., 
                             '2y': 2., 
                             '3y': 3., 
                             '5y': 5., 
                             '10y': 10., 
                             '15y': 15., 
                             '20y': 20., 
                             '30y': 30.}
    
    # Check whether bump size(s) and time(s) were passed and equal in dimension
    if bump_size is not None and bump_time is not None:
        
        # Check whether bump time was specified as one of the ISDA-SIMM tenors
        if type(bump_time) is str and bump_time.lower()[:9] == 'isda-simm':
            # Extract the tenor key from the bump time string and then obtain 
            # the bump time float from the ISDA-SIMM tenors dict and assign the 
            # corresponding curve vector index
            bump_time_tenor = bump_time.split()[1].lower()
            bump_time = ISDA_SIMM_tenors_dict[bump_time_tenor]
            bump_time_idx = int(bump_time*n_annual_trading_days)
            
            # Bump the yield curve at the specified bump time with the 
            # specified bump size
            bumped_time_0_R_curve[bump_time_idx] = (bumped_time_0_R_curve[bump_time_idx] 
                                                     + bump_size)
            
            ## Linear interpolation of the bump between the ISDA-SIMM tenors 
            ## immediately preceeding and following the specified bump time
            # Obtain the ISDA-SIMM tenors dict key index corresponding to the 
            # tenor key for determining the preceding and following ISDA-SIMM 
            # tenors
            dict_idxs = list(ISDA_SIMM_tenors_dict)
            idx = np.where(np.array(dict_idxs)==bump_time_tenor)[0][0]
            
            # Interpolate from the tenor immediately preceeding the bump time
            if bump_time_tenor != '2w':
                prev_time_tenor = dict_idxs[idx-1]
                prev_time = ISDA_SIMM_tenors_dict[prev_time_tenor]
                left_slope = bump_size/((bump_time - prev_time)*n_annual_trading_days)
                
                prev_time_idx = int(prev_time*n_annual_trading_days)
                for count, t_idx in enumerate(range(prev_time_idx, bump_time_idx)):
                    bumped_time_0_R_curve[t_idx] = (bumped_time_0_R_curve[t_idx] 
                                                     + left_slope*count)
            
            # Interpolate from the tenor immediately following the bump time
            if bump_time_tenor != '30y':
                next_time_tenor = dict_idxs[idx+1]
                next_time = ISDA_SIMM_tenors_dict[next_time_tenor]
                right_slope = bump_size/((bump_time - next_time)*n_annual_trading_days)
                
                next_time_idx = int(next_time*n_annual_trading_days)
                for count, t_idx in enumerate(range(bump_time_idx+1, next_time_idx+1)):
                    bumped_time_0_R_curve[t_idx] = (bumped_time_0_R_curve[t_idx] 
                                                     + bump_size 
                                                     + right_slope*(count+1))
    else:
        raise ValueError('no bump size or bump time was passed.')
        
    # Save time-zero zero rate curve data to Parquet file with current date 
    # and time in filename
    if verbose:
        print('\nSaving data in Parquet format with brotli compression...')
    file_dir_and_name = str(data_dir + 
                            'Bumped Time-Zero zero rate Curve'.replace(' ','_') 
                            + '-' + datetime.now().strftime('%Y-%m-%d_%H%M%S') 
                            + '.parquet')
    write_Parquet_data(experiment_dir, file_dir_and_name, bumped_time_0_R_curve)
    if verbose:
        print(f'\nBumped time-zero zero rate curve data saved to {file_dir_and_name}')
    
    # Plot time-zero zero rate curve and save figure with current date and 
    # time in filename
    if plot_curve == True:
        plot_title = 'Bumped Time-Zero Zero Rate Curve'
        x_label = 'Time $t$'
        y_label = '$R^M(0,t)$'
        
        if verbose:
            print('\nPlotting time-zero zero rate curve...')
        plot_time_series(experiment_dir, n_annual_trading_days, False, 
                         plot_title, bumped_time_0_R_curve, None, x_label, 
                         None, y_label, None)
    
    if verbose:
        print('Time-zero zero rate curve bump function terminated.\n\n')
    
    return bumped_time_0_R_curve
    
def construct_time_0_zero_bond_curve(experiment_dir: str,
                                     n_annual_trading_days: int,
                                     time_0_R_curve: list,
                                     bumped: bool = False,
                                     plot_curve: bool = False,
                                     spot_interest_rate_type: str = 'continuously-compounded',
                                     verbose: bool = False
                                    ) -> list:
    """
    Info:
        This function computes the market time-zero zero-bond curve P^M(0,t) 
        using the market time-zero zero rate curve R^M(0,t) as an input.
        
    Input:
        bumped: a bool specifying whether or not the curve is constructed from 
            a bumped time-zero zero rate curve.
        
        experiment_dir: the directory to which the results are saved.
        
        n_annual_trading_days: the number of trading days per year, used as the 
            time axis discretization.
        
        plot_curve: a bool specifying whether or not the resulting curve is 
            plotted and saved to the local folder.
            
        spot_interest_rate_type: a str specifying whether the time-zero zero-
            coupon curve consists of either continuously-compounded spot 
            interest rates ("continuously-compounded") or simply-compounded 
            spot interest rates ("simply-compounded").
            
        time_0_R_curve: an ndarray containing the time values of the time-zero 
            zero rate curve.
            
        verbose: a bool which if False blocks the function prints.
            
    Output:
        time_0_P_curve: an ndarray containing the time values of the market 
            time-zero zero-bond curve.
    """
    # Check whether Figures and Data directories located in current directory 
    # and if not, create them
    figures_dir = os.path.join(experiment_dir, 'Figures\\')
    data_dir = os.path.join(experiment_dir, 'Data\\')
    
    if not os.path.isdir(figures_dir):
        os.makedirs(figures_dir)
        
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    
    Delta_t = 1/n_annual_trading_days
    
    # Initialize the time-zero zero rate bond curve
    time_0_P_curve = np.zeros_like(time_0_R_curve)
    
    if spot_interest_rate_type.lower() == 'continuously-compounded':
        for time_t, R_0_t in enumerate(time_0_R_curve):
            time_0_P_curve[time_t] = np.exp(-R_0_t*time_t*Delta_t)
    elif spot_interest_rate_type.lower() == 'simply-compounded':
        for idx in range(len(time_0_P_curve)):
            time_0_P_curve[time_t] = (1/(1 + R_0_t*time_t*Delta_t))
    else:
        raise ValueError('spot interest rate type of time-zero zero rate' 
                         + ' curve not recognized. Enter as' 
                         + ' "continuously-compounded" or "simply-compounded".')
    
    # Save time-zero zero-bond curve data to Parquet file with current date 
    # and time in filename
    if verbose:
        print('\nSaving data in Parquet format with brotli compression...')
    file_dir_and_name = str(data_dir 
                            + 'Time-Zero Zero-Bond Curve'.replace(' ','_') + '-' 
                            + datetime.now().strftime('%Y-%m-%d_%H%M%S') 
                            +'.parquet')
    write_Parquet_data(experiment_dir, file_dir_and_name, time_0_P_curve)
    
    if verbose:
        print(f'Zero-bond curve data saved to {file_dir_and_name}')
    
    # Plot zero-bond curve and save image
    if plot_curve == True:
        plot_title = ('Time-Zero Zero-Bond Curve' if not bumped 
                      else 'Bumped Time-Zero Zero-Bond Curve')
        x_label = 'Time $t$'
        y_label = '$P^M(0,t)$'
        
        if verbose:
            print('\nPlotting zero-bond curve...')
        plot_time_series(experiment_dir, n_annual_trading_days, False, 
                         plot_title, time_0_P_curve, None, x_label, None, 
                         y_label, None)
    
    if verbose:
        print('\nTime-zero zero-bond curve function terminated.\n\n')
    
    return time_0_P_curve


def construct_time_0_inst_forw_rate_curve(experiment_dir: str,
                                          n_annual_trading_days: int,
                                          time_0_P_curve: int,
                                          time_0_rate: float,
                                          bumped: bool = False,
                                          plot_curve: bool = False,
                                          verbose: bool = False
                                         ) -> list:
    """
    Info:
        This function computes the market time-zero instantaneous forward rate 
            curve f^M(0,t) using the market time-zero zero rate bond curve as 
            input.
        
    Input:
        bumped: a bool specifying whether or not the curve is constructed from 
            a bumped time-zero zero rate curve.
        
        experiment_dir: the directory to which the results are saved.
        
        n_annual_trading_days: the number of trading days per year, used as the 
            time axis discretization.
        
        plot_curve: a bool specifying whether or not the resulting curve is 
            plotted and saved to the local folder.
            
        time_0_P_curve: an ndarray containing the market time-zero zero-bond 
            curve from which the market time-zero instantaneous forward rate 
            curve is constructed.
            
        time_0_rate: the initial interest rate on which the term structure is 
            based.
            
        verbose: a bool which if False blocks the function prints.
        
    Output:
        time_0_f_curve: an ndarray containing the time values of the market 
            time-zero instantaneous forward rate curve.
    """
    # Check whether Figures and Data directories located in current directory 
    # and if not, create them
    figures_dir = os.path.join(experiment_dir, 'Figures/')
    data_dir = os.path.join(experiment_dir, 'Data\\')
    
    if not os.path.isdir(figures_dir):
        os.makedirs(figures_dir)
        
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    
    n_trading_days = len(time_0_P_curve) - 1
    delta_T = 1/n_annual_trading_days
    time_0_f_curve = np.zeros_like(time_0_P_curve)
    log_time_0_P_curve = np.log(time_0_P_curve)
    
    for index in range(n_trading_days):
        slope = (log_time_0_P_curve[index+1] 
                 - log_time_0_P_curve[index])/delta_T 
        time_0_f_curve[index] = -slope
        
    # Imply last change as half the linear continuation of last slope
    time_0_f_curve[-1] = (time_0_f_curve[-2] + (time_0_f_curve[-2] 
                                                - time_0_f_curve[-3])/2)
    
    # Save time-zero instantaneous forward curve data to Parquet file with 
    # current date and time in filename
    if verbose:
        print('\nSaving data in Parquet format with brotli compression...')
    file_dir_and_name = str(data_dir 
                            +  'Time-Zero Instantaneous Forward Rate Curve'.replace(' ','_') 
                            + '-' + datetime.now().strftime('%Y-%m-%d_%H%M%S') 
                            + '.parquet')
    write_Parquet_data(experiment_dir, file_dir_and_name, time_0_f_curve)
    
    if verbose:
        print(f'Time-zero instantaneous forward rate curve curve data saved to {file_dir_and_name}')
    
    # Plot instantaneous forward curve and its derivative and save images
    if plot_curve == True:
        plot_title = ('Time-Zero Instantaneous Forward Rate Curve' if not bumped 
                      else 'Bumped Time-Zero Instantaneous Forward Rate Curve')
        x_label = 'Time $t$'
        y_label = '$f^M(0,t)$'
        
        if verbose:
            print('\nPlotting instantaneous forward rate curve...')
            
        if bumped is True:
            # plot_title = plot_title + '\nClose-Up'
            plot_time_series(experiment_dir, n_annual_trading_days, False, 
                          plot_title, time_0_f_curve, None, x_label, None, 
                          y_label, None)
        
        else:
            plot_time_series(experiment_dir, n_annual_trading_days, False, 
                          plot_title, time_0_f_curve, None, x_label, None, 
                          y_label, y_limits=[time_0_rate*.95, time_0_rate*1.05])
        
    if verbose:
        print('\nTime-zero instantaneous forward rate curve function terminated.\n\n')
    
    return time_0_f_curve

def construct_time_0_inst_forw_rate_curve_derivative(experiment_dir: str,
                                                     n_annual_trading_days: int,
                                                     time_0_f_curve: list,
                                                     bumped: bool = False,
                                                     plot_curve: bool = False,
                                                     verbose: bool = False
                                                    ) -> list:
    """
    Info:
        This function approximates the derivative of the market time-zero 
        instantaneous forward rate curve with respect to its maturities.
            
        
    Input:
        bumped: a bool specifying whether or not the curve is constructed from 
            a bumped time-zero zero rate curve.
            
        experiment_dir: the directory to which the results are saved.
        
        n_annual_trading_days: the number of trading days per year, used as the 
            time axis discretization.
        
        plot_curve: a bool specifying whether or not the resulting curve is 
            plotted and saved to the local folder.
            
        time_0_f_curve: an ndarray containing the market time-zero 
            instantaneous forward rate curve from which the derivative curve is 
            created.
            
        verbose: a bool which if False blocks the function prints.
        
    Output:
        time_0_dfdt_curve: an ndarray containing the time derivative values of 
            the time-zero instantaneous forward rate curve.
    """
    # Check whether Figures and Data directories located in current directory 
    # and if not, create them
    figures_dir = os.path.join(experiment_dir, 'Figures/')
    data_dir = os.path.join(experiment_dir, 'Data\\')
    
    if not os.path.isdir(figures_dir):
        os.makedirs(figures_dir)
        
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    
    n_trading_days = len(time_0_f_curve) - 1
    Delta_t = 1/len(time_0_f_curve)
    
    time_0_dfdt_curve = np.zeros(n_trading_days)
    for time_t in range(n_trading_days):
        time_0_dfdt_curve[time_t] = (time_0_f_curve[time_t+1] 
                                     - time_0_f_curve[time_t])/Delta_t
        
    # For the last value, it is sufficient to assume it stays constant
    time_0_dfdt_curve[-1] = time_0_dfdt_curve[-2]
    
    # Save time-zero instantaneous forward curve data to Parquet file with 
    # current date and time in filename
    if verbose:
        print('\nSaving data in Parquet format with brotli compression...')
    file_dir_and_name = str(data_dir + 
                            'Time-Zero Instantaneous Forward Rate Curve'.replace(' ','_') 
                            + '-' + datetime.now().strftime('%Y-%m-%d_%H%M%S') 
                            + '.parquet')
    write_Parquet_data(experiment_dir, file_dir_and_name, time_0_dfdt_curve)
    if verbose:
        print(f'Time-zero instantaneous forward rate curve curve data saved to {file_dir_and_name}')
    
    # Plot instantaneous forward curve and its derivative and save images
    if plot_curve == True:
        plot_title = (('Partial Derivative With Respect to Maturity of' 
                       + '\nTime-Zero Instantaneous Forward Rate Curve') if not bumped 
                      else ('Bumped Partial Derivative With Respect to Maturity of' 
                            + '\nTime-Zero Instantaneous Forward Rate Curve'))
        x_label = 'Time $t$'
        y_label = '$\partial f^M(0,t)/\partial T$'
        
        if verbose:
            print('\nPlotting partial derivative with respect to maturity of' 
              + 'instantaneous forward rate curve...')
        plot_time_series(experiment_dir, n_annual_trading_days, False, 
                         plot_title, time_0_dfdt_curve, None, x_label, None, 
                         y_label, None)
    
    if verbose:
        print('\nTime-zero instantaneous forward rate curve function terminated.\n\n')
    
    return time_0_dfdt_curve