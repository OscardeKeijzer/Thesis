# -*- coding: utf-8 -*-

# This module provides classes for the implementation of the one-factor 
# Hull-White short rate model and the pricing of various derivatives including 
# swaps, European swaptions, and Bermudan swaptions. ...

# References:
#     [1] Brigo, D., & Mercurio, F. (2006). Interest Rate Models - Theory and 
#         Practice: With Smile, Inflation, and Credit. Springer, Berlin, 
#         Heidelberg. doi: 10.1007/978-3-540-34604-3
#     [2] Glasserman, P. (2003). Monte Carlo Methods in Financial Engineering. 
#         Springer. doi: 10.1007/978-0-387-21617-1
#     [3] Hoencamp, J., Jain, S., & Kandhai, D. (2022). A Semi-Static 
#         Replication Approach to Efficient Hedging and Pricing of Callable IR 
#         Derivatives. arXiv. doi: 10.48550/arXiv.2202.01027

# Imports
import copy
from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as so
import scipy.stats as st
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow.keras as keras

# Local imports
from bonds_and_bond_options import (B_func,
                                    construct_zero_coupon_bonds_curve, 
                                    price_coupon_bearing_bond, 
                                    price_coupon_bearing_bond_option, 
                                    price_zero_coupon_bond, 
                                    price_zero_coupon_bond_forward_exact,
                                    price_zero_coupon_bond_option_exact,
                                    price_zero_coupon_bond_option_MC)
from curve_functions import (bump_time_0_zero_coupon_curve, 
                             construct_time_0_inst_forw_rate_curve, 
                             construct_time_0_inst_forw_rate_curve_derivative, 
                             construct_time_0_zero_coupon_curve, 
                             construct_time_0_zero_bond_curve)
from data_functions import (read_Parquet_data, 
                            write_Parquet_data)
from neural_networks import ( 
                             ShallowFeedForwardNeuralNetwork)
from interest_rate_functions import (eval_annuity_terms,
                                     eval_cont_comp_spot_rate,
                                     eval_simply_comp_forward_rate,
                                     eval_swap_rate,
                                     interpolate_zero_rate)
from least_squares_Monte_Carlo import (price_Bermudan_stock_option_LSM, 
                                       price_Bermudan_swaption_LSM_Q)
from one_factor_Hull_White_model import (eval_discount_factors, 
                                         one_factor_Hull_White_exp_Q, 
                                         one_factor_Hull_White_var, 
                                         gen_one_factor_Hull_White_paths)
from plotting_functions import (plot_time_series, 
                                plot_one_factor_Hull_White_histogram, 
                                visualize_neural_network)
from swaps import (eval_moneyness_adjusted_fixed_rate, 
                   price_forward_start_swap)
from swaptions import (eval_approx_vol, 
                       most_expensive_European, 
                       price_European_swaption_exact_Bachelier, 
                       price_European_swaption_exact_Jamshidian, 
                       price_European_swaption_MC_Q)

###
class OneFactorHullWhiteModel:
    """
    This class represents an instance of the one-factor Hull & White short rate 
    model with methods for the construction of the fundamental interest rate 
    curves, the simulation of the short rates, and the bumping of the yield 
    curve under ISDA-SIMM specifications.
    
    Attributes:
        experiment_dir_name: the name of the directory to which the results are 
            saved.
        
        n_annual_trading_days: an int specifying the number of trading days 
            per year for the time axis discretization.
        
        time_horizon: the time in years for which the market time-zero 
            zero rate curve will be constructed.
        
        time_0_rate: the time-zero interest rate on which the market time-zero 
            zero rate curve is based.
            
    Methods:
        construct_curves(): constructs the fundamental interest rate curves.
        
        read_input_Parquet_data(): read previously simulated data stored in 
            Parquet format.
        
        sim_short_rate_paths(): simulate the short rate process paths 
            under the risk-neutral measure Q.
            
        plot_short_rate_paths(): plot histogram and evolution of the short rate 
            process over time.
        
        log_short_rate_paths(): create a text file containing computational and 
            statistical information on the short rate process simulation.
            
        bump_yield_curve: bump the yield curve and construct the resulting 
            zero-coupon bond and instantaneous forward rate curves for use in 
            the bump-and-revalue method under the ISDA-SIMM specifications.
            
        resim_short_rate_paths(): resimulate a new set of short rate paths from 
            the bumped interest rate curves.
    """
    
    def __init__(self,
                 experiment_dir_name: str,
                 n_annual_trading_days: int,
                 time_horizon: float,
                 time_0_rate: float
                ) -> object:
        self.n_annual_trading_days = n_annual_trading_days
        self.time_horizon = time_horizon
        self.time_0_rate = time_0_rate
        
        ## Create directory for saving the experiments that use the fundamental 
        # interest rate curves constructed in this method.
        # Store instance time for directory and file names
        os.chdir(os.getcwd())
        instance_starting_time = datetime.now().strftime('%Y-%m-%d_%H%M')
        print('One-factor Hull-White model instance initialized at ' 
              + f'{instance_starting_time}')
        
        # Assign directories for script, results, inputs
        script_dir = os.path.dirname(os.path.realpath(__file__))
        results_dir = os.path.join(script_dir, 'Results\\')
        input_dir = os.path.join(script_dir, 'Inputs\\')
        
        # Check whether results directory was created
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        # Create experiment directory
        if experiment_dir_name is not None:
            experiment_dir = os.path.join(results_dir, experiment_dir_name)
        else:
            experiment_dir = os.path.join(results_dir, 'Experiment '  
                                          + f'{instance_starting_time}.')

        if not os.path.isdir(experiment_dir):
            os.makedirs(experiment_dir)
            
        # Assign instance variables for the experiment result and input 
        # directories
        self.results_dir = results_dir
        self.input_dir = input_dir
        self.experiment_dir = experiment_dir
        
        # Check whether data directory located in current experiment directory and 
        # if not, create data directory
        self.data_dir = os.path.join(self.experiment_dir, 'Data\\')
    
        if not os.path.isdir(self.data_dir):
            os.makedirs(self.data_dir)
            
        # Check whether Figures directory located in current directory and if not, 
        # create Figures directory
        self.figures_dir = os.path.join(self.experiment_dir, 'Figures\\')
    
        if not os.path.isdir(self.figures_dir):
            os.makedirs(self.figures_dir)

        # Instantiate the ISDA-SIMM risk tenor dictionary
        self.ISDA_SIMM_tenors_dict = {'2w': 1/26, 
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

    def construct_curves(self, 
                         plot_curve: bool = False,
                         shape: str = 'flat',
                         spot_interest_rate_type: str = 'continuously-compounded',
                         std_dev: float = 0.,
                         verbose: bool = False
                        ):
        """
        Info: this method constructs the fundamental interest rate curves, 
            i.e., the market time-zero zero rate curve, the market 
            time-zero zero-coupon bond curve, and the market time-zero 
            instantaneous forward rate curve and its derivative with respect to 
            maturity.
            
        Input: 
            plot_curve: plot_curve: a bool specifying whether or not the 
                resulting curves are plotted and saved to the local folder.
                
            shape: a str specifying what the shape of the time-zero zero rate 
                curve should be. 'flat' creates a flat curve, 'normal' creates 
                an increasing curve, and 'inverted' creates a decreasing curve.
                
            spot_interest_rate_type: a str specifying whether the time-zero 
                zero-coupon curve consists of either continuously-compounded 
                spot interest rates ("continuously-compounded") or 
                simply-compounded spot interest rates ("simply-compounded").
                
            std_dev: the size of the standard deviation in the random number 
                vector that is generated in order to construct the zero rate 
                curve.
                
            verbose: a bool which if False blocks the function prints.
        
        Output:
            self.time_0_R_curve: an ndarray containing the time values of the 
                time-zero zero rate curve.
                
            self.time_0_P_curve: an ndarray containing the time values of the 
                market time-zero zero-bond curve.
                
            self.time_0_f_curve: an ndarray containing the time values of the 
                market time-zero instantaneous forward curve.
                
            self.time_0_dfdt_curve: an ndarray containing the time derivative 
                values of the time-zero instantaneous forward curve.
        """
        self.plot_curves = plot_curve
        self.shape = shape
        self.spot_interest_rate_type = spot_interest_rate_type
        self.std_dev = std_dev
        
        self.time_0_R_curve = construct_time_0_zero_coupon_curve(
                                        self.experiment_dir, 
                                        self.n_annual_trading_days, 
                                        self.time_horizon, self.time_0_rate, 
                                        plot_curve, shape, std_dev, verbose)

        self.time_0_P_curve = construct_time_0_zero_bond_curve(
                                        self.experiment_dir, 
                                        self.n_annual_trading_days, 
                                        self.time_0_R_curve, False, plot_curve, 
                                        spot_interest_rate_type, verbose)

        self.time_0_f_curve = construct_time_0_inst_forw_rate_curve(
                                        self.experiment_dir, 
                                        self.n_annual_trading_days, 
                                        self.time_0_P_curve, self.time_0_rate, 
                                        False, plot_curve, verbose)

        self.time_0_dfdt_curve = construct_time_0_inst_forw_rate_curve_derivative(
                                        self.experiment_dir, 
                                        self.n_annual_trading_days, 
                                        self.time_0_f_curve, False, plot_curve,
                                        verbose)
        
    def read_input_short_rate_paths_Parquet(self,
                                            file_name: str
                                           ):
        self.input_file_name_and_dir = self.input_dir + file_name
        self.r_t_paths = read_Parquet_data(self.input_file_name_and_dir)
        
        
    def sim_short_rate_paths(self,
                             a_param: float,
                             n_paths: int,
                             r_t_sim_time: float,
                             sigma: float,
                             antithetic: bool = True,
                             seed: int = None,
                             r_t_process_type: str = 'zero-mean',
                             sim_type: str = 'Euler', 
                             verbose: bool = False
                            ):
        """
        Info: this method simulates a desired number of short rates under the 
            risk-neutral measure using exact simulation or Euler-Maruyama 
            discretization by simulating the short rates directly or through 
            use of the zero-mean process.
            
        Input:
            a_param: the constant mean reversion rate parameter in the 1F-HW 
                model.
                
            antithetic: if True, the zero-mean process is simulated using 
                antithetic standard normal random draws. Note: this was 
                implemented for the shifted zero-mean process simulation using 
                the Euler method only.
                
            n_paths:the number of short rate paths to be simulated.
            
            r_t_process_type: if 'direct', the short rates are 
                simulated directly; if 'zero-mean', the short rates are 
                simulated using the zero-mean process.
            
            r_t_sim_time: the simulation time in years.
            
            sigma: the constant volatility factor of the short rates.
            
            sim_type: a str specifying the simulation discretization type: if 
                'Euler', the short rates are simulated using Euler-Maruyama 
                discretization; if 'exact', the short rates are simulated by 
                sampling the exact distribution.
                
            time_0_rate: ...
                
            seed: the seed for the randon number generator used for the generation 
                of the short rate paths.
        
        Output:
            r_t_paths: a 2D ndarray containing the simulated short rate paths 
                along a number of columns corresponding to the number of paths 
                and a number of rows being a discrete short rate time series of 
                length equal to the total number of trading days.
                
            x_t_paths: if the short rates were simulated as a shifted zero-mean 
                process: a 2D ndarray containing the zero-mean process paths 
                along a number of columns corresponding to the number of paths 
                and a number of rows being a discrete short rate time series of 
                length equal to the total number of trading days.
        """
        self.antithetic = antithetic
        self.a_param = a_param
        self.n_paths = n_paths
        self.seed = seed
        self.r_t_process_type = r_t_process_type
        self.r_t_sim_time = r_t_sim_time
        self.sigma = sigma
        self.sim_type = sim_type
        
        if self.r_t_sim_time > self.time_horizon:
            raise ValueError('short rate simulation time exceeds simulated ' 
                             + 'time of fundamental interest rate curves.')
            
        # Store simulation starting time for timing purposes
        self.r_t_sim_start_time = datetime.now()
        
        ## Simulate short rates based on user specifications
        self.short_rate_paths_output = gen_one_factor_Hull_White_paths(a_param, 
                                            antithetic, self.experiment_dir, 
                                            self.n_annual_trading_days, 
                                            n_paths, r_t_process_type, 
                                            seed, sigma, r_t_sim_time, 
                                            sim_type, 
                                            self.time_0_dfdt_curve, 
                                            self.time_0_f_curve, 
                                            self.time_0_rate, verbose)
        self.r_t_paths = (self.short_rate_paths_output 
                          if self.r_t_process_type.lower() == 'direct'
                          else self.short_rate_paths_output[0])
        self.x_t_paths = (None if self.r_t_process_type.lower() == 'direct'
                          else self.short_rate_paths_output[1])
            
        # Evaluate short rate simulation time
        self.r_t_sim_runtime = (datetime.now() 
                                - self.r_t_sim_start_time).total_seconds()
            
    def plot_short_rate_paths(self,
                              plot_time: float,
                              plot_type: str
                             ):
        """
        Info: this method generates a histogram of the simulated short rate 
            paths over time and, if selected, a plot of the either the mean 
            short rate, the individual short rates, or both, over time.
            
        Input: 
            plot_time: the time at which the histogram and short rate paths are 
                plotted.
                
            plot_type: a str specifying what is plotted: if 'individual', the 
                individual short rate paths are plotted; if 'mean', the mean 
                short rate path is plotted; if 'both', both the individual and 
                mean short rate paths are plotted; if None, only the histogram 
                is plotted.
        """
        # If no plotting time was specified, the final simulation time will be 
        # used
        if plot_time is None:
            self.plot_time = self.r_t_sim_time
        else:
            self.plot_time = plot_time
        
        expectation = one_factor_Hull_White_exp_Q(self.a_param, 
                                self.n_annual_trading_days, self.r_t_paths, 
                                self.sigma, 0., plot_time, self.time_0_f_curve)
        expected_variance = one_factor_Hull_White_var(self.a_param, self.sigma, 
                                                      0, plot_time)
        
        if self.sim_type.lower() == 'euler':
            plot_title = '1F-HW Short Rate Paths Under $\mathbb{Q}$\nEuler Method'
            if self.antithetic:
                plot_title = plot_title + ' - Antithetic Paths'
        elif self.sim_type.lower() == 'exact':
            plot_title = '1F-HW Short Rate Paths Under $\mathbb{Q}$\nExact Simulation'
        x_label = 'Time $t$'
        y_label = 'Short Rate $r_t$'
        
        # Plot histogram of short rate values at the plotting time
        print('\nPlotting histogram of short rate values at ' 
              + f'time {self.plot_time}...')
        hist_plot_start_time = datetime.now()
        plot_one_factor_Hull_White_histogram(self.experiment_dir, expectation, 
                                             self.n_annual_trading_days, 
                                             plot_title, plot_time, 
                                             self.r_t_paths, expected_variance, 
                                             'Short Rate $r_t$')
        self.hist_plot_time = (datetime.now() 
                               - hist_plot_start_time).total_seconds()
        
        if plot_type.lower() == 'individual' or plot_type.lower() == 'both':
            # Plot individual short rate paths over time
            print('\nPlotting individual short rate paths...')
            plot_ind_paths_start_time = datetime.now()
            plot_time_series(self.experiment_dir, self.n_annual_trading_days, 
                             False, plot_title, self.r_t_paths, None, x_label, 
                             None, y_label, None)
            
            self.individual_plot_time = (datetime.now() 
                                         - plot_ind_paths_start_time).total_seconds()
        
        
        if plot_type.lower() == 'mean' or plot_type.lower() == 'both':
            # Plot mean short rate path over time
            print('\nPlotting mean short rate path...')
            plot_mean_start_time = datetime.now()
            plot_time_series(self.experiment_dir, self.n_annual_trading_days, 
                             True, plot_title, self.r_t_paths, None, x_label, 
                             None, y_label, None)
            
            plot_title = plot_title + ' - Zoomed Out'
            plot_time_series(self.experiment_dir, self.n_annual_trading_days, 
                             True, plot_title, self.r_t_paths, None, x_label, 
                             None, y_label, 
                             [np.min(self.r_t_paths), np.max(self.r_t_paths)])
            
            self.mean_plot_time = (datetime.now() 
                                   - plot_mean_start_time).total_seconds()
            
    def log_short_rate_paths(self):
        """
        Info: this method creates a .txt text file containing statistical and 
            computational information on the short rate simulation.
        """
        # Check whether data directory located in current experiment directory and 
        # if not, create data directory
        self.data_dir = os.path.join(self.experiment_dir, 'Data\\')
    
        if not os.path.isdir(self.data_dir):
            os.makedirs(self.data_dir)
            
        # Compute statistical parameters of short rates and zero-mean process
        self.r_t_mean = np.mean(self.r_t_paths[-1])
        self.r_t_var = np.var(self.r_t_paths[-1])
        if self.x_t_paths is not None:
            self.x_t_mean = np.mean(self.x_t_paths[-1])
            self.x_t_var = np.var(self.x_t_paths[-1])
            
        expectation = one_factor_Hull_White_exp_Q(self.a_param, 
                                self.time_0_f_curve, self.r_t_paths, 
                                self.n_annual_trading_days, self.sigma, 0, 
                                self.short_rate_sim_time)
        expected_variance = one_factor_Hull_White_var(self.a_param, self.sigma, 
                                                0, self.short_rate_sim_time)
        
        # Save output data to Parquet file with current date and time in 
        # filename
        if self.antithetic:
            file_dir_and_name_r_t = str(self.data_dir + 
                                    '1F-HW_short_rate_paths_array' 
                                    + f'_{self.r_t_process_type}' 
                                    + f'_{self.sim_type}_antithetic-' 
                                    + datetime.now().strftime('%Y-%m-%d_%H%M%S')
                                    + '.parquet')
            file_dir_and_name_x_t = str(self.data_dir + 
                                    '1F-HW_zero_mean_process_paths_array' 
                                    + f'_{self.r_t_process_type}' 
                                    + f'_{self.sim_type}_antithetic-' 
                                    + datetime.now().strftime('%Y-%m-%d_%H%M%S')
                                    + '.parquet')
            write_Parquet_data(self.experiment_dir, file_dir_and_name_r_t, 
                               self.r_t_paths)
            write_Parquet_data(self.experiment_dir, file_dir_and_name_x_t, 
                               self.x_t_paths)
        else:
            file_dir_and_name_r_t = str(self.data_dir + 
                                    '1F-HW_short_rate_paths_array' 
                                    + f'_{self.r_t_process_type}' 
                                    + f'_{self.sim_type}-' 
                                    + datetime.now().strftime('%Y-%m-%d_%H%M%S')
                                    + '.parquet')
            write_Parquet_data(self.experiment_dir, file_dir_and_name_r_t, 
                               self.r_t_paths)
            if self.x_t_paths is not None:
                file_dir_and_name_x_t = str(self.data_dir + 
                                        '1F-HW_zero_mean_process_paths_array' 
                                        + f'_{self.r_t_process_type}' 
                                        + f'_{self.sim_type}-' 
                                        + datetime.now().strftime('%Y-%m-%d_%H%M%S')
                                        + '.parquet')
                write_Parquet_data(self.experiment_dir, file_dir_and_name_x_t, 
                                   self.x_t_paths)
        
        print(f'\nShort rate data was saved to {file_dir_and_name_x_t}' 
              + ' with brotli compression.')
        
        ## Save log file of statistical and computational details if experiment 
        ## directory specified
        if self.antithetic:
            with open(self.data_dir + '1F-HW_short_rates_Q' 
                      + f'_{self.r_t_process_type}' + f'_{self.sim_type}'
                      + '_antithetic_Log.txt', 'a') as f:
                f.write('\n' + 10*'*' + ' Parameters ' + 10*'*')
                f.write(f'\na parameter: {self.a_param}\n' 
                        + f'number of paths: {self.n_paths}\n'
                        + f'short rate process type: {self.r_t_process_type}\n' 
                        + f'short rate simulation time: {self.short_rate_sim_time}\n'
                        + f'sigma: {self.sigma}\n' 
                        + f'short rate simulation type: {self.sim_type}\n')
                f.write('\n' + 10*'*' + ' Statistical Details ' + 10*'*')
                f.write(f'\nThe mean value {self.r_t_mean} at time' 
                        + f' {self.short_rate_sim_time} deviates' 
                        + f' {(self.r_t_mean-expectation)/expectation*100:.4f}%' 
                        + f' from the expectation {expectation}.')
                f.write(f'\nThe variance {self.r_t_var} at time' 
                        + f' {self.short_rate_sim_time} deviates' 
                        + f' {(self.r_t_var-expected_variance)/expected_variance*100:.4f}%' 
                        + f' from the expected variance {expected_variance}.\n\n')
                
                f.write(10*'*' + ' Computational Details ' + 10*'*')
                f.write('\n1F-HW short rate simulation function runtime:' 
                        + f' {self.short_rate_sim_runtime} seconds.\n')
                if hasattr(self, 'hist_plot_time'):
                    f.write('1F-HW short rate histogram plotting time:' 
                            + f' {self.hist_plot_time} seconds.\n')
                if hasattr(self, 'individual_plot_time'):
                    f.write('1F-HW short rate individual paths plotting time:' 
                            + f' {self.individual_plot_time} seconds.\n')
                if hasattr(self, 'mean_plot_time'):
                    f.write('1F-HW short rate mean paths plotting time:' 
                            + f' {self.mean_plot_time} seconds.\n')
        else:
            with open(self.data_dir + '1F-HW_short_rates_Q' 
                      + f'_{self.r_t_process_type}' + f'_{self.sim_type}'
                      + '_Log.txt', 'a') as f:
                f.write('\n' + 10*'*' + ' Parameters ' + 10*'*')
                f.write(f'\na parameter: {self.a_param}\n' 
                        + f'number of paths: {self.n_paths}\n'
                        + f'short rate process type: {self.r_t_process_type}\n' 
                        + f'short rate simulation time: {self.short_rate_sim_time}\n'
                        + f'sigma: {self.sigma}\n' 
                        + f'short rate simulation type: {self.sim_type}\n')
                f.write('\n' + 10*'*' + ' Statistical Details ' + 10*'*')
                f.write(f'\nThe mean value {self.r_t_mean} at time' 
                        + f' {self.short_rate_sim_time} deviates' 
                        + f' {(self.r_t_mean-expectation)/expectation*100:.4f}%' 
                        + f' from the expectation {expectation}.')
                f.write(f'\nThe variance {self.r_t_var} at time' 
                        + f' {self.short_rate_sim_time} deviates' 
                        + f' {(self.r_t_var-expected_variance)/expected_variance*100:.4f}%' 
                        + f' from the expected variance {expected_variance}.\n\n')
                
                f.write(10*'*' + ' Computational Details ' + 10*'*')
                f.write('\n1F-HW short rate simulation function runtime:' 
                        + f' {self.short_rate_sim_runtime} seconds.\n')
                if hasattr(self, 'hist_plot_time'):
                    f.write('1F-HW short rate histogram plotting time:' 
                            + f' {self.hist_plot_time} seconds.\n')
                if hasattr(self, 'individual_plot_time'):
                    f.write('1F-HW short rate individual paths plotting time:' 
                            + f' {self.individual_plot_time} seconds.\n')
                if hasattr(self, 'mean_plot_time'):
                    f.write('1F-HW short rate mean paths plotting time:' 
                            + f' {self.mean_plot_time} seconds.\n')
            
        return self.r_t_mean, self.r_t_var, self.short_rate_sim_runtime
    
    def bump_yield_curve(self,
                         bump_time: float,
                         bump_size: float = .0001,
                         plot_curve: bool = False,
                         verbose: bool = False
                        ):
        """
        Info: this method copies the original yield curve and then bumps the 
            copy at a specified time with a specified bump size for the 
            bump-and-revalue method.
            
        Input:
            bump_size: bump_size: the size of the bump for use in the 
                resimulation of a bumped set of short rates in the 
                bump-and-revalue sensitivity estimation.
                
            bump_time: the time of the zero rate curve bump for use in the
                resimulation of a bumped set of short rates in the 
                bump-and-revalue sensitivity estimation method. Can be entered 
                in years as type float or as a string specifying one of the 
                ISDA-SIMM tenors {2w, 1m, 3m, 6m, 1Y, 2Y, 3Y, 5Y, 10Y, 15Y, 
                20Y, 30Y} formatted as "ISDA-SIMM {tenor}".
                
            plot_curve: a bool specifying whether or not the resulting curves 
                are Plotted and saved to the local folder.
                
            verbose: a bool which if False blocks the function prints.
            
        Output:
            self.time_0_R_curve_bumped: an ndarray containing the time values 
                of the bumped time-zero zero rate curve.
                
            self.time_0_P_curve: an ndarray containing the time values of the 
                bumped market time-zero zero-bond curve.
                
            self.time_0_f_curve: an ndarray containing the time values of the 
                bumped market time-zero instantaneous forward rate curve.
                
            self.time_0_dfdt_curve: an ndarray containing the time derivative 
                bumped values of the time-zero instantaneous forward rate 
                curve.
        """
        self.bump_size = bump_size
        self.bump_time = bump_time
        
        self.time_0_R_curve_bumped = bump_time_0_zero_coupon_curve(
                                                bump_time, self.experiment_dir, 
                                                self.n_annual_trading_days, 
                                                plot_curve, 
                                                self.time_0_R_curve, bump_size, 
                                                verbose)
        
        self.time_0_P_curve_bumped = construct_time_0_zero_bond_curve(
                                                self.experiment_dir, 
                                                self.n_annual_trading_days, 
                                                self.time_0_R_curve_bumped, 
                                                True, plot_curve, 
                                                self.spot_interest_rate_type, 
                                                verbose)
        
        self.time_0_f_curve_bumped = construct_time_0_inst_forw_rate_curve(
                                                self.experiment_dir, 
                                                self.n_annual_trading_days, 
                                                self.time_0_P_curve_bumped, 
                                                self.time_0_rate, True, 
                                                plot_curve, verbose)
        
        self.time_0_dfdt_curve_bumped = construct_time_0_inst_forw_rate_curve_derivative(
                                                self.experiment_dir, 
                                                self.n_annual_trading_days, 
                                                self.time_0_f_curve_bumped, 
                                                True, plot_curve, verbose)
    
    def resim_short_rates(self,
                          seed: int, 
                          verbose: bool = False
                         ):
        """
        Info: this method simulates a new set of short rates for use in the 
            bump-and-revalue sensitivity estimation method.
            
        Input: 
            seed: an int specifying the seed for the randon number generator 
                used for the generation of the short rate paths.
                
            verbose: a bool which if False blocks the function prints.
            
        Output:
            r_t_paths_resim: a 2D ndarray containing the resimulated short rate 
                paths along a number of columns corresponding to the number of 
                paths and a number of rows being a discrete short rate time 
                series of length equal to the total number of trading days.
                
            x_t_paths_resim: if the short rates were resimulated as a shifted 
                zero-mean process: a 2D ndarray containing the zero-mean 
                process paths along a number of columns corresponding to the 
                number of paths and a number of rows being a discrete short 
                rate time series of length equal to the total number of trading 
                days.
        """
        # Check whether interest rate curves were bumped
        if not hasattr(self, 'time_0_R_curve_bumped'):
            raise ValueError('yield curve has not been bumped yet. Use the ' 
                             + 'bump_yield_curve() method before resimulating ' 
                             + 'short rates.')
            
        # Check whether seed matches that of the original simulation
        if (self.seed is None):
            raise ValueError('original short rates were not simulated using a ' 
                             + 'user-specified RNG seed. Use the ' 
                             + 'sim_short_rate_paths() method and be sure to ' 
                             + 'pass the same seeds.')
        elif (isinstance(self.seed, float) and self.seed != seed):
            print('short rate resimulation seed does not match the original ' 
                  + f'seed {self.seed}. Resimulating short rates using ' 
                  + 'original seed.')
            seed = self.seed
        
        # Simulate a new set of short rates
        self.short_rate_paths_output_resim = gen_one_factor_Hull_White_paths(
                                                self.a_param, self.antithetic, 
                                                self.experiment_dir, 
                                                self.n_annual_trading_days, 
                                                self.n_paths, 
                                                self.r_t_process_type, seed, 
                                                self.sigma, self.r_t_sim_time, 
                                                self.sim_type, 
                                                self.time_0_dfdt_curve_bumped, 
                                                self.time_0_f_curve_bumped, 
                                                self.time_0_rate, verbose)
        self.r_t_paths_resim = (self.short_rate_paths_output_resim 
                                if self.r_t_process_type.lower() == 'direct'
                                else self.short_rate_paths_output_resim[0])
        self.x_t_paths_resim = (None if self.r_t_process_type.lower() == 'direct'
                                else self.short_rate_paths_output_resim[1])
        
class BermudanSwaptionLSM:
    """
    This class represents a Bermudan forward start interest swap option 
    approximated by the least-squares Monte Carlo (LSM) method. 
    
    Attributes: 
        model: an instance of the OneFactorHullWhiteModel class containing the 
            fundamental interest rate curves and simulated short rates.
            
    Methods:
        price_Bermudan_swaption_LSM_Q(): computes the LSM price of the Bermudan 
            swaption.
            
        eval_forward_sensitivities_bump_and_revalue_ISDA_SIMM(): computes the 
            forward sensitivities of the Bermudan swaption using the 
            bump-and-revalue method according to the ISDA-SIMM specifications.
            
        eval_most_expensive_European(): evaluates the tenor structure of the 
            most expensive European swaption whose tenor structure is spanned 
            by the Bermudan tenor structure along with the price difference 
            between the two.
    """
    
    def __init__(self,
                 model: object,
                ) -> object:
        self.model = model
        
    def price_Bermudan_swaption_LSM_Q(self,
                                      degree: int,
                                      fixed_rate: float,
                                      moneyness: float,
                                      notional: float,
                                      swaption_type: str,
                                      tenor_structure: list,
                                      time_t: float,
                                      units_basis_points: bool,
                                      payoff_var: str = 'swap',
                                      plot_regression: bool = False,
                                      plot_timeline: bool = False,
                                      regression_series: str = 'power',
                                      r_t_paths: list = None,
                                      verbose: bool = False,
                                      x_t_paths: list = None
                                     ):
        """
        Info: this method values a Bermudan forward start interest rate swap 
            option within the one-factor Hull-White model under the 
            risk-neutral measure using the least-squares Monte Carlo 
            approach.
            
        Input:
            degree: an int specifying the degree of the polynomial used in the 
                regression.
                
            fixed_rate: a float specifying the fixed interest rate used as the 
                strike rate of the (underlying) interest rate swap.
                
            moneyness: a float specifying the level of moneyness e.g. 1.0 
                equals 100% moneyness.
                
            notional: a float specifying the notional amount of the 
                (underlying) interest rate swap.
            
            payoff_var: a str specifying the (underlying) swap payoff function: 
                if 'swap', the swap payoff is determined using the forward swap 
                rate; if 'forward', the swap payoff is determined using the 
                simply compounded forward rate.
                
            plot_regression: a bool which if True plots the regression at a 
                select number of intermediate time steps.
                
            plot_timeline: a bool specifying whether or not the (underlying)
                swap timeline is plotted and saved to the local folder.
                
            r_t_paths: a 2D ndarray containing the simulated short rate paths 
                along a number of columns corresponding to the number of paths 
                and a number of rows being a discrete short rate time series of 
                length equal to the total number of trading days.
            
            swaption_type: a str specifying the swaption type which can be 
                'payer' or 'receiver'.
                
            tenor_structure: a list containing the (underlying) forward swap's 
                starting date as the first entry and the payment dates as the 
                remaining entries.
                
            time_t: a float specifying the time of evaluation in years.
            
            units_basis_points: a bool which if True causes the output value to 
                be given in basis points of the notional.
                
            verbose: a bool which if False blocks the function prints.

            x_t_paths: a 2D ndarray containing the zero-mean process paths 
                along a number of columns corresponding to the number of paths 
                and a number of rows being a discrete short rate time series of 
                length equal to the total number of trading days.
                
        Output:
            self.pathwise_LSM_prices: a 1D ndarray containing the pathwise LSM 
                prices of the Bermudan swaption.
                
            self.live_paths_array: a 2D ndarray containing the live status of 
                the Monte Carlo paths on the monitor dates.
                
            self.pathwise_stopping_times: 2 2D ndarray containing the pathwise 
                stopping times of the Bermudan swaption.
                
            self.mean_LSM_price: the mean of the pathwise LSM prices.
            
            self.se_LSM_price: the standard error of the pathwise LSM prices.
        """
        self.degree = degree
        self.fixed_rate = fixed_rate
        self.notional = notional
        self.payoff_var = payoff_var
        self.plot_regression = plot_regression
        self.plot_timeline = plot_timeline
        self.regression_series = regression_series
        self.moneyness = moneyness
        self.swaption_type = swaption_type
        self.tenor_structure = tenor_structure
        self.time_t = time_t
        self.units_basis_points = units_basis_points
        
        if r_t_paths is None:
            r_t_paths = self.model.r_t_paths
        if x_t_paths is None:
            x_t_paths = self.model.x_t_paths
        
        if moneyness is not None:
            self.fixed_rate = eval_moneyness_adjusted_fixed_rate(moneyness, 
                                            self.model.n_annual_trading_days, 
                                            swaption_type, tenor_structure, 
                                            self.model.time_0_P_curve)
        
        (self.pathwise_LSM_prices, 
         self.live_paths_array, 
         self.pathwise_stopping_times) = price_Bermudan_swaption_LSM_Q(
                                                self.model.a_param, 
                                                degree, 
                                                self.model.experiment_dir, 
                                                fixed_rate, moneyness, 
                                                self.model.n_annual_trading_days, 
                                                notional, payoff_var,  
                                                plot_regression, plot_timeline, 
                                                regression_series, r_t_paths, 
                                                self.model.sigma, 
                                                swaption_type, tenor_structure, 
                                                time_t, 
                                                self.model.time_0_dfdt_curve, 
                                                self.model.time_0_f_curve, 
                                                self.model.time_0_P_curve, 
                                                units_basis_points, x_t_paths, 
                                                verbose)
        
        self.mean_LSM_price = np.mean(self.pathwise_LSM_prices)
        self.se_LSM_price = st.sem(self.pathwise_LSM_prices)
        
    def eval_forward_sensitivities_bump_and_revalue_ISDA_SIMM(self, 
                                                       bump_time: float, 
                                                       eval_time: float, 
                                                       bump_size: float = .0001, 
                                                       plot_bumped_curve: bool = False, 
                                                       bump_model: object = None,
                                                       r_t_paths: list = None,
                                                       seed: int = None,
                                                       verbose: bool = False,
                                                       x_t_paths: list = None
                                                      ):
        """
        Info: this method bumps the yield curve at one of the ISDA-SIMM tenors 
            and revalues the LSM price for the resulting bumped short rates, 
            then determines the sensitivity as the finite difference between 
            the bumped and unbumped short rates multiplied by the bump size 
            in accordance with the ISDA-SIMM specifications.
            
        Input:
            bump_model: an instance of the OneFactorHullWhite class that has 
                had its yield curve bumped and short rates resimulated.
            
            bump_size: the size of the bump for use in the resimulation of a 
                bumped set of short rates in the bump-and-revalue sensitivity 
                estimation.
                
            bump_time: the time of the zero rate curve bump for use in the
                resimulation of a bumped set of short rates in the 
                bump-and-revalue sensitivity estimation method. Can be entered 
                in years as type float or as a string specifying one of the 
                ISDA-SIMM tenors {2w, 1m, 3m, 6m, 1Y, 2Y, 3Y, 5Y, 10Y, 15Y, 
                20Y, 30Y} formatted as "ISDA-SIMM {tenor}".
                
            eval_time: the evaluation time of the sensitivity in years.
            
            plot_bumped_curve: a bool specifying whether or not the newly 
                bumped yield curve is to be plotted.
                
            seed: an integer specifying the RNG seed for the short rate 
                simulation. Must only be entered if the original short rates 
                are known not to have been simulated with a user-specified 
                seed.
                
            r_t_paths: a 2D ndarray containing the simulated short rate paths 
                along a number of columns corresponding to the number of paths 
                and a number of rows being a discrete short rate time series of 
                length equal to the total number of trading days.
                
            verbose: a bool which if False blocks the function prints.
                
            x_t_paths: a 2D ndarray containing the zero-mean process paths 
                along a number of columns corresponding to the number of paths 
                and a number of rows being a discrete short rate time series of 
                length equal to the total number of trading days.
        
        Output:
            self.pathwise_forward_sensitivities: a 1D ndarray containing the 
                pathwise sensitivites of the Bermudan swaption.
                
            self.mean_forward_sensitivity: the mean of the pathwise 
                sensitivities.
                
            self.se_forward_sensitivities: the standard error of the pathwise 
                sensitivities.
        """
        self.bump_size = bump_size
        self.bump_time = bump_time
        self.plot_bumped_curve = plot_bumped_curve
        self.seed = seed
        
        # Check whether Bermudan price was already evaluated
        if (not hasattr(self, 'pathwise_LSM_prices') 
            or self.time_t != eval_time):
            print('LSM price not yet evaluated. Now pricing Bermudan swaption...')
            self.price_Bermudan_swaption_LSM_Q(self.model.a_param, self.degree, 
                                               self.model.experiment_dir, 
                                               self.fixed_rate, self.moneyness, 
                                               self.model.n_annual_trading_days, 
                                               self.notional, self.payoff_var,  
                                               self.plot_regression, 
                                               self.plot_timeline, 
                                               self.regression_series, 
                                               r_t_paths, self.model.sigma, 
                                               self.swaption_type, 
                                               self.tenor_structure, eval_time, 
                                               self.model.time_0_dfdt_curve, 
                                               self.model.time_0_f_curve, 
                                               self.model.time_0_P_curve, 
                                               self.units_basis_points, 
                                               x_t_paths, verbose)
        
        if type(bump_time) is str and bump_time.lower()[:9] != 'isda-simm':
            raise ValueError('bump time was not passed as one of the ISDA-' 
                             + 'SIMM tenors. Enter as "ISDA-SIMM" followed by ' 
                             + 'a space and one of the following:\n' 
                             + f'{self.model.ISDA_SIMM_tenors_dict.keys()}')
        
        bump_time_tenor = bump_time.split()[1].lower()
        if bump_time_tenor not in self.model.ISDA_SIMM_tenors_dict.keys():
            raise ValueError('bump time was not passed as one of the ISDA-' 
                             + 'SIMM tenors. Enter as a string starting with ' 
                             + '"ISDA-SIMM" followed by a space and one of the' 
                             + ' following keys:\n\t' 
                             + f'{list(self.model.ISDA_SIMM_tenors_dict.keys())}')
        bump_time = self.model.ISDA_SIMM_tenors_dict[bump_time_tenor]
        
        # Check whether a bumped 1F-HW model was passed or whether one should 
        # be created
        if bump_model is None:
            # Make a deep copy of the 1F-HW instance for bumping without 
            # altering the original instance's properties
            self.bump_model = copy.deepcopy(self.model)
            
            # Check whether previous model short rates were simulated using 
            # user-specified RNG seed
            if self.bump_model.seed is None and self.seed is not None:
                if verbose:
                    print('the original short rates were not simulated ' 
                          + 'with a user-specified RNG seed. Now simulating ' 
                          + f'passed model\'s short rates with seed {self.seed}.')
                    
                self.bump_model.seed = self.seed
                self.bump_model.sim_short_rate_paths(self.bump_model.a_param, 
                                self.bump_model.n_paths,
                                self.bump_model.r_t_sim_time, 
                                self.bump_model.sigma, 
                                self.bump_model.antithetic, self.seed, 
                                self.bump_model.r_t_process_type, 
                                self.bump_model.sim_type, verbose)
                
            elif (self.bump_model.seed is not None 
                  and self.bump_model.seed != self.seed):
                if verbose:
                    print('the specified RNG seed does not match the seed of ' 
                          + 'the original short rates. Now using the ' 
                          + f'passed model\'s seed {bump_model.seed}.')
                self.seed = self.bump_model.seed
            
            
            # Bump the yield curve
            self.bump_model.bump_yield_curve(self.bump_size, self.bump_time, 
                                              self.plot_bumped_curve)
            self.bump_model.resim_short_rates(self.bump_model.time_0_f_curve_bumped, 
                                    self.bump_model.time_0_dfdt_curve_bumped, 
                                    self.bump_model.seed)
            
        elif (hasattr(bump_model, 'time_0_R_curve_bumped') 
              and bump_model.seed == self.seed 
              and bump_model.n_paths == self.model.n_paths):
            self.bump_model = bump_model
            
        else:
            raise ValueError('Passed bumped model seed does not correspond.')
        
        # Revalue the swaption
        self.bump_model.time_0_R_curve = self.bump_model.time_0_R_curve_bumped
        self.bump_model.time_0_f_curve = self.bump_model.time_0_f_curve_bumped
        self.bump_model.time_0_dfdt_curve = self.bump_model.time_0_dfdt_curve_bumped
        self.bump_model.time_0_P_curve = self.bump_model.time_0_P_curve_bumped
        self.bump_model.r_t_paths = self.bump_model.r_t_paths_resim
        self.bump_model.x_t_paths = self.bump_model.x_t_paths_resim
        self.LSM_revalue_inst = BermudanSwaptionLSM(self.bump_model)
        self.LSM_revalue_inst.price_Bermudan_swaption_LSM_Q(self.degree, 
                                        self.fixed_rate, None, 
                                        self.notional, self.swaption_type, 
                                        self.tenor_structure, eval_time, 
                                        self.units_basis_points, 
                                        self.payoff_var, self.plot_regression, 
                                        self.plot_timeline,  
                                        self.regression_series, 
                                        self.bump_model.r_t_paths, 
                                        verbose, self.bump_model.x_t_paths)
        self.pathwise_LSM_prices_reval = self.LSM_revalue_inst.pathwise_LSM_prices
        self.mean_LSM_price_reval = self.LSM_revalue_inst.mean_LSM_price
        self.se_LSM_price_reval = self.LSM_revalue_inst.se_LSM_price
        
        # Evaluate the pathwise sensitivities
        self.pathwise_forward_sensitivities = ((self.pathwise_LSM_prices_reval 
                                        - self.pathwise_LSM_prices))
        self.mean_forward_sensitivity = np.mean(self.pathwise_forward_sensitivities)
        self.se_forward_sensitivities = st.sem(self.pathwise_forward_sensitivities)
        
    def eval_most_expensive_European(self,
                                     verbose
                                     ):
        """
        Info: this method evaluates the most expensive European tenor structure 
            and the size of the gap between the most expensive European 
            swaption price and the Bermudan swaption price.
            
        Input:
            verbose: a bool which if False blocks the function prints.
            
        Output:
            self.MEE_gap: the difference between the Bermudan swaption price 
                and the most expensive European swaption price whose tenor 
                structure is spanned by the tenor structure of the Bermudan.
                
            self.MEE_tenor_structure: the tenor structure of the most expensive 
                European swaption.
        """
        if not hasattr(self, 'mean_LSM_price'):
            raise ValueError('Bermudan LSM price not yet computed.')
            
        self.MEE_gap, self.MEE_tenor_structure = most_expensive_European(
                                                    self.model.a_param, 
                                                    self.mean_LSM_price, 
                                                    self.fixed_rate, None, 
                                                    self.model.n_annual_trading_days, 
                                                    self.notional, 
                                                    self.model.sigma, 
                                                    self.swaption_type, 
                                                    self.tenor_structure, 
                                                    self.time_t, 
                                                    self.model.time_0_f_curve, 
                                                    self.model.time_0_P_curve, 
                                                    self.model.time_0_rate, 
                                                    self.units_basis_points)
        
class BermudanSwaptionRLNN:
    """
    This class represents a Bermudan forward start interest rate swap option 
    under the one-factor Hull-White (1F-HW) short rate model as approximated by 
    the regress-later neural network (RLNN) method. 
    
    Info: The Bermudan swaption is valued within the 1F-HW model under the 
        risk-neutral measure Q using the RLNN method. A portfolio of 
        zero-coupon bond options or forwards is used for the semi-static 
        replication of the Bermudan swaption.
        
    Attributes:
        fixed_rate: the fixed rate of the Bermudan swaption.
        
        model: the one-factor Hull-White model instance containing the main 
            model parameters, fundamental interest rate curves, and simulated 
            short rate paths.
            
        notional: the notional of the Bermudan swaption.
        
        swaption_type: the Bermudan swaption type which can be 'payer' or 
            'receiver'.
            
        tenor_structure: a list containing the underlying forward swap's tenor 
            structure i.e. the starting date as the first entry and the payment 
            dates as the remaining entries.
            
        time_t: the time of valuation of the Bermudan swaption.
            
        input_dim: the dimensions of the input layer.
    
    Methods:
        ...
            
    """
    
    def __init__(self,
                 fixed_rate: float,
                 model: object,
                 notional: float,
                 moneyness: bool,
                 swaption_type: str,
                 tenor_structure: list,
                 n_run : int = None,
                 units_basis_points: bool = False
                ) -> object:
        
        # Assign main parameters from constructor input
        self.fixed_rate = fixed_rate
        self.model = model
        self.notional  = notional
        self.moneyness = moneyness
        self.swaption_type = swaption_type
        self.tenor_structure = (tenor_structure if isinstance(tenor_structure, 
                                                              np.ndarray)
                                else np.array(tenor_structure))
        self.n_run = n_run
        self.units_basis_points = units_basis_points
        
        # Assign secondary parameters
        self.expiry = self.tenor_structure[0]
        self.monitor_dates = self.tenor_structure[:-1]
        self.maturity = self.tenor_structure[-1]
        self.tenor = len(self.tenor_structure)
        self.index = 0
        self.tenor_structure_notation = (f'{round(self.tenor_structure[0])}Y' 
                                            + f'x{round(self.tenor_structure[-1] - self.tenor_structure[0])}Y')
        
        # Adjust the fixed rate if a level of moneyness was specified
        if moneyness is not None:
            self.fixed_rate = eval_moneyness_adjusted_fixed_rate(moneyness, 
                                            self.model.n_annual_trading_days, 
                                            swaption_type, tenor_structure, 
                                            self.model.time_0_P_curve)
        
        # Initialize name of directory for storing neural network weights
        self.weights_dir = (self.model.data_dir + 
                            f'\\NN_weights_{self.tenor_structure_notation}Y' 
                            + f'_N_={self.model.n_paths}' 
                            + f'_moneyness={self.moneyness}')
        
    def replicate(self,
                  neural_network: object,
                  time_t: float,
                  batch_size: int = 32,
                  input_dim: int = 1,
                  learn_rate: float = .0003,
                  n_epochs: int = 2000,
                  n_hidden_nodes: int = 64,
                  save_weights: bool = False,
                  seed_biases: int = None,
                  seed_weights: int = None,
                  test_fit: bool = False,
                  train_size: float = .2
                 ):
        """
        Info: this method calls the training and portfolio construction methods 
            in order to replicate the Bermudan swaption value over its tenor 
            structure.
        """
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.learn_rate = learn_rate
        self.n_epochs = n_epochs
        self.n_hidden_nodes = n_hidden_nodes
        self.save_weights = save_weights
        self.seed_biases = seed_biases
        self.seed_weights = seed_weights
        self.test_fit = test_fit
        self.time_t = time_t
        self.train_size = train_size
        
        # Update weights directory name and create it
        self.weights_dir = (self.weights_dir 
                            + f'_learn_rate={np.format_float_positional(self.learn_rate, trim="-")}'
                            + f'_n_epochs={self.n_epochs}'
                            + f'_n_hidden_nodes={self.n_hidden_nodes}'
                            + f'_train_size={self.n_hidden_nodes}')
        if not os.path.isdir(self.weights_dir):
            os.makedirs(self.weights_dir)
        
        self.ZBO_portfolio_weights_list = []
        self.ZBO_portfolio_strikes_list = []
        
        self.MSE = np.zeros(len(self.monitor_dates))
        
        ## Replication of the Bermudan swaption using zero-coupon bond options
        # Iterate backward over the monitor dates
        for count, monitor_date in enumerate(self.tenor_structure[-2::-1]):
            t_monitor_idx = int(monitor_date*self.model.n_annual_trading_days)
            t_idx_tenor = len(self.tenor_structure) - 2 - count
            
            # Select the zero-mean process values as input for the neural 
            # network and then start the training procedure
            x_t_in = self.model.x_t_paths[t_monitor_idx]
            self.train(monitor_date, None, t_idx_tenor, t_monitor_idx, x_t_in, 
                       batch_size=self.batch_size, learn_rate=self.learn_rate, 
                       n_epochs=self.n_epochs, 
                       n_hidden_nodes=self.n_hidden_nodes, 
                       save_weights=self.save_weights, test_fit=self.test_fit, 
                       train_size=self.train_size)
            
            # Construct the replicating portfolio at the current monitor date
            self.construct_ZBO_portfolio()
            
    def train(self,
              monitor_date: float,
              neural_network: object,
              t_idx_tenor: float,
              t_monitor_idx: float,
              x_t_input: list,
              batch_size: int = 32,
              learn_rate: int = .0003,
              n_epochs: int = 4500,
              n_hidden_nodes: int = 64,
              save_weights: bool = False,
              test_fit: bool = False,
              train_size: float = .2
             ):
        """
        Info: this method prepares the data for training and initializes and 
            fits the neural network.
        """
        ## Split input zero-mean process data into training and test sets and 
        ## evaluate the current exercise value
        # Split the input data Monte Carlo path indices into training and test 
        # sets
        if not hasattr(self, 'x_train_idxs'):
            print('Splitting inputs into training and test sets...')
            indices = np.arange(self.model.n_paths)
            self.x_train_idxs, self.x_test_idxs = train_test_split(indices, 
                                                        test_size=1-self.train_size, 
                                                        random_state=True)
            self.n_train = len(self.x_train_idxs)
            self.n_test = len(self.x_test_idxs)
          
        # Assign the current training and test sets from the provided input 
        # zero-mean process values
        self.x_train = x_t_input[self.x_train_idxs]
        self.x_test = x_t_input[self.x_test_idxs]
        
                                                         
        ## Determine pathwise exercise and continuation values
        # Evaluate exercise value at current monitor date
        V_exercise = np.maximum(price_forward_start_swap(
                                self.model.a_param, self.fixed_rate, 
                                self.model.n_annual_trading_days, 
                                self.notional, 'swap', False, 
                                self.model.r_t_paths[:,self.x_train_idxs], 
                                self.model.sigma, self.swaption_type, 
                                self.tenor_structure[t_idx_tenor:], 
                                monitor_date, self.model.time_0_f_curve, 
                                self.model.time_0_P_curve, False, 
                                self.model.x_t_paths[:,self.x_train_idxs]), 
                                0.)
        
        # Evaluate the discounted continuation values that correspond to 
        # holding the Bermudan swaption at the current monitor date until the 
        # following monitor date
        # Initialize the continuation values array
        V_continuation = np.zeros_like(V_exercise)
        
        # Select the corresponding short rate paths for discounting the 
        # continuation values
        self.r_t_train = self.model.r_t_paths[:,self.x_train_idxs]
        
        # For each hidden node, determine the continuation value by evaluating 
        # the portfolio of zero-coupon bond options with weights and strikes 
        # as determined in the previous step using the current realizations of 
        # the zero-mean processes. For the final monitor date, the Bermudan 
        # swaption expires worthless and so the replicating portfolio values 
        # are set to zero.
        
        if monitor_date < self.tenor_structure[-2]:
            ZBO_portfolio = np.zeros((len(self.current_ZBO_portfolio_weights), 
                                      len(self.x_train)))
            
            for count, idx in enumerate(self.positive_strikes_idxs):
                strike = self.current_ZBO_portfolio_strikes[count]
                weight = self.current_ZBO_portfolio_weights[count]
                if self.swaption_type.lower() == 'payer':
                    option_type = 'call'
                elif self.swaption_type.lower() == 'receiver':
                    option_type = 'put'
                
                ZBO_pathwise_prices = price_zero_coupon_bond_option_exact(
                                            self.model.a_param, self.maturity, 
                                            self.tenor_structure[t_idx_tenor+1], 
                                            self.model.n_annual_trading_days, 
                                            option_type, None, self.model.sigma, 
                                            strike, monitor_date, self.model.time_0_f_curve, 
                                            self.model.time_0_P_curve, False, 
                                            self.model.x_t_paths[:,self.x_train_idxs], 
                                            None)
                    
                ZBO_portfolio[count] = weight*ZBO_pathwise_prices
                                        
            V_continuation = np.maximum(np.sum(ZBO_portfolio, axis=0), 0.)

        # Evaluate and scale the y data
        self.y_train = np.maximum(V_exercise, V_continuation)
        self.y_train_scaling_factor = np.mean(self.y_train[self.y_train>0.])
        self.y_train_rescaled = self.y_train/self.y_train_scaling_factor
        
        ## Training and fitting the neural network
        # Evaluate the input zero-coupon bond prices
        self.z_in = price_zero_coupon_bond(self.model.a_param, 
                                        self.model.n_annual_trading_days, 
                                        self.model.r_t_paths[t_monitor_idx,
                                                             self.x_train_idxs], 
                                        self.model.sigma, monitor_date, 
                                        self.maturity, self.model.time_0_f_curve, 
                                        self.model.time_0_P_curve, 
                                        self.x_train)
        self.z_in_scale_b = np.mean(self.z_in)
        self.z_in_scale_w = np.std(self.z_in)
        self.z_in_rescaled = (self.z_in - self.z_in_scale_b)/self.z_in_scale_w
        
        # Initialize and fit the neural network
        if neural_network:
            self.neural_network = neural_network
        else:
            self.neural_network_obj = ShallowFeedForwardNeuralNetwork(
                                            self.swaption_type, 
                                            n_hidden_nodes=self.n_hidden_nodes, 
                                            seed_biases=self.seed_biases, 
                                            seed_weights=self.seed_weights)
            self.neural_network = self.neural_network_obj.neural_network
        
        optimizer_Adamax = keras.optimizers.Adamax(learning_rate=self.learn_rate)
        self.neural_network.compile(loss='MSE', optimizer=optimizer_Adamax)
        
        if (self.save_weights is not None and self.index > 0):
            print('Loading weights...')
            weight_name = self.weights_dir + '\\Bermudan NN Weights' + str(self.index-1) + '.h5'
            if self.n_run is not None:
                weight_name = weight_name + 'n_run_' + str(self.n_run)
            self.neural_network.load_weights(weight_name)
        
        callback_early_stopping = keras.callbacks.EarlyStopping(monitor='loss', 
                                                        mode='min', verbose=True, 
                                                        patience=100, 
                                                        restore_best_weights=True)
        
        print(f'Fitting neural network at monitor date {monitor_date}...')
        self.neural_network.fit(self.z_in_rescaled, self.y_train_rescaled, 
                                epochs=self.n_epochs, verbose=False, 
                                callbacks=[callback_early_stopping], 
                                batch_size=self.batch_size)
        self.neural_network.summary()
        
        # Save the MSE
        y_true = self.y_train_rescaled
        P_in = price_zero_coupon_bond(self.model.a_param, 
                                        self.model.n_annual_trading_days, 
                                        self.model.r_t_paths[t_monitor_idx,
                                                             self.x_train_idxs], 
                                        self.model.sigma, 
                                        monitor_date, self.maturity, 
                                        self.model.time_0_f_curve, 
                                        self.model.time_0_P_curve, 
                                        self.x_train)
        x_train_scaled = (P_in - self.z_in_scale_b)/self.z_in_scale_w
        y_pred = self.neural_network.predict(x_train_scaled).flatten()
        self.MSE[t_idx_tenor] = np.mean((y_true - y_pred)**2)
        
        if self.test_fit:
            P_in = price_zero_coupon_bond(self.model.a_param, 
                                    self.model.n_annual_trading_days, 
                                    self.model.r_t_paths[t_monitor_idx,
                                                         self.x_test_idxs], 
                                    self.model.sigma, monitor_date, 
                                    self.maturity, self.model.time_0_f_curve, 
                                    self.model.time_0_P_curve, self.x_test)
            x_test_scaled = (P_in - self.z_in_scale_b)/self.z_in_scale_w
            y_test = self.neural_network.predict(x_test_scaled).flatten()
            plt.scatter(self.x_test, y_test, color='r', 
                        label='Neural Network Test Data Prediction')
            plt.scatter(self.x_train, y_true, label='Training Data', s=1, 
                        zorder=10)
            plt.scatter(self.x_train, y_pred, label='Training Data Prediction', 
                        color='blue', alpha=.5)
            plt.xlabel('$x_t$')
            plt.title(f'{self.tenor_structure_notation} Bermudan ' 
                      + f'{self.swaption_type.capitalize()} Swaption' 
                      + f'\nNeural Network Fit at $T$ = {monitor_date}')
            plt.legend()
            plt.show()    
        
        if self.save_weights is not None:
            print('Saving weights...')
            weight_name = self.weights_dir + '\\Bermudan NN Weights' + str(self.index) + '.h5'
            name2 = self.weights_dir + '\\Bermudan NN Nodes' + str(self.n_hidden_nodes) + '1F' + str(self.index) + '.h5'
            if self.n_run is not None:
                weight_name = weight_name + 'n_run_' + str(self.n_run)
                name2 = name2 + 'n_run' + str(self.n_run)
            self.neural_network.save_weights(weight_name)
            self.neural_network.save_weights(name2)
            self.index += 1
            
    def construct_ZBO_portfolio(self):
        """
        Info: this method constructs a portfolio of zero-coupon bond options 
            that replicates the Bermudan swaption at a given monitor date.
        """
        # Determine zero-coupon bond option payoff type as call or put 
        # corresponding to swaption type (receiver or payer)
        if self.swaption_type.lower() == 'receiver':
            delta_payoff = -1
        elif self.swaption_type.lower() == 'payer':
            delta_payoff = 1
        
        # Obtain the weights and biases of the hidden layer and the weights of 
        # the output layer
        self.weights_hidden = self.neural_network.layers[0].get_weights()[0].flatten()
        self.biases_hidden = self.neural_network.layers[0].get_weights()[1]
        self.weights_output = self.neural_network.layers[1].get_weights()[0].flatten()
        
        # Determine the portfolio strikes
        self.weights_hidden_unscaled = (delta_payoff*self.weights_hidden
                                        /self.z_in_scale_w)
        
        portfolio_strikes = np.zeros(self.n_hidden_nodes)
        for idx in range(self.n_hidden_nodes):
            portfolio_strikes[idx] = (-1*delta_payoff*self.biases_hidden[idx]
                                      /self.weights_hidden_unscaled[idx] 
                                      + self.z_in_scale_b)
        
        # Determine the portfolio weights
        portfolio_weights = (self.weights_output*self.weights_hidden_unscaled
                              *self.y_train_scaling_factor)
        
        # Select the final weights and strikes for the portfolio
        self.positive_strikes_idxs = np.where(portfolio_strikes > 0.)[0]
        final_portfolio_weights = portfolio_weights[self.positive_strikes_idxs]
        final_portfolio_strikes = portfolio_strikes[self.positive_strikes_idxs]
        
        # Store the portfolio weights and strikes
        self.current_ZBO_portfolio_weights = final_portfolio_weights
        self.current_ZBO_portfolio_strikes = final_portfolio_strikes
        
        # Save ZBO portfolio weights and strikes in their respective lists
        self.ZBO_portfolio_weights_list.insert(0, self.current_ZBO_portfolio_weights)
        self.ZBO_portfolio_strikes_list.insert(0, self.current_ZBO_portfolio_strikes)
        
    def price_direct_estimator(self,
                               time_t: float, 
                               r_t_paths: list = None,
                               x_t_paths: list = None
                              ):
        """
        Info: this method evaluates the direct price estimator of the Bermudan 
            swaption using the previously constructed portfolio of zero-coupon 
            bond options.
        """
        if time_t > self.tenor_structure[-2]:
            raise ValueError('Bermudan swaption evaluation time occurs after last monitor date.')
        else:
            self.pricing_time = time_t
            
        ## Price the replicating portfolio
        if self.swaption_type.lower() == 'payer':
            option_type = 'call'
        elif self.swaption_type.lower() == 'receiver':
            option_type = 'put'    
        
        # If no zero-mean process paths were passed, the test paths of the 
        # corresponding replicating portfolio as split during training will be 
        # used for the direct price estimate
        if x_t_paths is None:
            self.pricing_r_t_paths = self.model.r_t_paths[:,self.x_test_idxs]
            self.pricing_x_t_paths = self.model.x_t_paths[:,self.x_test_idxs]
        elif x_t_paths is not None and r_t_paths is not None:
            self.pricing_r_t_paths = r_t_paths
            self.pricing_x_t_paths = x_t_paths    
        
        # Select the replicating portfolio that corresponds to the pricing time
        n_paths = np.shape(self.pricing_x_t_paths)[1]
        (portfolio_date, 
         self.pricing_portfolio, 
         strikes, weights)          = self.select_portfolio(time_t, n_paths)
            
        # Price the Bermudan swaption using the replicating portfolio
        for count, strike in enumerate(strikes):
            weight = weights[count]
            ZBO_pathwise_prices = price_zero_coupon_bond_option_exact(
                                        self.model.a_param, self.maturity, 
                                        portfolio_date, 
                                        self.model.n_annual_trading_days, 
                                        option_type, None, self.model.sigma, 
                                        strike, time_t, 
                                        self.model.time_0_f_curve, 
                                        self.model.time_0_P_curve, False, 
                                        self.pricing_x_t_paths, None)
            
            self.pricing_portfolio[count] = (weight*ZBO_pathwise_prices)
            
        self.direct_price_estimator = (np.sum(self.pricing_portfolio, 
                                                        axis=0)
                                       *10**4/self.notional 
                                       if self.units_basis_points 
                                       else np.sum(self.pricing_portfolio, 
                                                              axis=0))
        self.mean_direct_price_estimator = np.mean(self.direct_price_estimator)
        self.se_direct_price_estimator = st.sem(self.direct_price_estimator)
        
    def eval_EPE_profile(self,
                       compare: bool,
                       eval_times: list = None,
                       r_t_paths: list = None,
                       units_basis_points: bool = False,
                       x_t_paths: list = None
                      ):
        """
        Info: this method evaluates the expected positive exposure (EPE) of the 
            Bermudan swaption using the replicating portfolio.
        """
        ## Check whether specific evaluation times were passed
        if eval_times is not None:
            if not isinstance(eval_times, np.ndarray):
                eval_times = np.array([eval_times])
                
            if eval_times[-1] <= self.tenor_structure[-2]:
                self.EPE_eval_times = eval_times
            else:
                raise ValueError('the passed exposure evaluation times may ' 
                                  + 'not surpass the last monitor date of ' 
                                  + f'{self.tenor_structure[-2]} years.')
        # If not, evaluate the EPE biweekly from time zero until the last monitor 
        # date
        else:
            self.EPE_eval_times = np.linspace(0., self.tenor_structure[-2], 
                                              num=24*int(self.tenor_structure[-2])+1)
            
        ## Evaluate the exposures
        if r_t_paths is not None and x_t_paths is not None:
            self.EPE_r_t_paths = r_t_paths
            self.EPE_x_t_paths = x_t_paths
        else:
            self.EPE_r_t_paths = self.model.r_t_paths[:,self.x_test_idxs]
            self.EPE_x_t_paths = self.model.x_t_paths[:,self.x_test_idxs]
        n_paths = np.shape(self.EPE_x_t_paths)[1]
        n_eval_times = len(self.EPE_eval_times)
            
        # Evaluate pathwise stopping times
        self.eval_exercise_strategies(self.EPE_r_t_paths, self.EPE_x_t_paths)
            
        # Initialize exposures matrix and mean exposures vector
        self.EPE_RLNN = np.zeros((n_eval_times, n_paths))
        self.mean_EPE_RLNN = np.zeros(n_eval_times)
        
        
        if compare:
            LSM_instance = BermudanSwaptionLSM(self.model)
            self.EPE_LSM = np.zeros_like(self.EPE_RLNN)
            self.mean_EPE_LSM = np.zeros_like(self.mean_EPE_RLNN)
        
        # Loop over EPE evaluation times and evaluate the exposures
        for count, eval_time in enumerate(self.EPE_eval_times):
            # print(f'eval_time: {eval_time}')
            tenor_idx = np.searchsorted(self.tenor_structure, eval_time)
            print(f'tenor_idx: {tenor_idx}')
            
            # print(f'self.live_paths_array[tenor_idx]: {self.live_paths_array[tenor_idx]}')
            # print(f'non-zero self.live_paths_array[tenor_idx]: {np.nonzero(self.live_paths_array[tenor_idx])}')
            # print(f'length of non-zero self.live_paths_array[tenor_idx]: {len(np.nonzero(self.live_paths_array[tenor_idx])[0])}')
            # print(f'mean live_paths_array: {np.mean(self.live_paths_array[tenor_idx])}')
            
            # # Method 1:
            # if eval_time <= self.expiry:
            #     self.price_direct_estimator(eval_time, self.EPE_r_t_paths, self.EPE_x_t_paths)
            # else:
            #     self.price_direct_estimator(self.tenor_structure[tenor_idx], self.EPE_r_t_paths, self.EPE_x_t_paths)
            # discount_factors = eval_discount_factors(
            #                                 self.model.n_annual_trading_days, 
            #                                 self.EPE_r_t_paths, 0., 
            #                                 self.tenor_structure[tenor_idx])#eval_time)
            # # print(f'mean discount_factors: {np.mean(discount_factors)}')
            # self.EPE_RLNN[count] = (discount_factors
            #                         *self.live_paths_array[tenor_idx]
            #                         *np.maximum(self.direct_price_estimator, 
            #                                     0.))
            
            # Method 2:
            discount_factors = eval_discount_factors(
                                            self.model.n_annual_trading_days, 
                                            self.EPE_r_t_paths, 0., eval_time)
            self.price_direct_estimator(eval_time, self.EPE_r_t_paths, 
                                        self.EPE_x_t_paths)
            self.EPE_RLNN[count] = (discount_factors
                                    *self.live_paths_array[tenor_idx]
                                    *np.maximum(self.direct_price_estimator, 
                                                0.))
            
            # # Method 3:
            # if eval_time <= self.expiry:
            #     self.price_direct_estimator(eval_time, self.EPE_r_t_paths, self.EPE_x_t_paths)
            # else:
            #     self.price_direct_estimator(self.tenor_structure[tenor_idx-1], self.EPE_r_t_paths, self.EPE_x_t_paths)
            # discount_factors = eval_discount_factors(
            #                                 self.model.n_annual_trading_days, 
            #                                 self.EPE_r_t_paths, 0., 
            #                                 self.tenor_structure[tenor_idx])#eval_time)
            # # print(f'mean discount_factors: {np.mean(discount_factors)}')
            # self.EPE_RLNN[count] = (discount_factors
            #                         *self.live_paths_array[tenor_idx]
            #                         *np.maximum(self.direct_price_estimator, 
            #                                     0.))
            
            # print(f'shape(exposures[count): {np.shape(self.EPE_RLNN[count])}')
            # print(f'np.shape(self.live_paths_array[tenor_idx]): {np.shape(self.live_paths_array[tenor_idx])}')
            # print(f'np.shape(self.direct_price_estimator): {np.shape(self.direct_price_estimator)}')
            # self.EPE_RLNN[count] = self.direct_price_estimator
            # print(f'mean price: {np.mean(self.direct_price_estimator)}')
            self.mean_EPE_RLNN[count] = np.mean(self.EPE_RLNN[count])
            # self.mean_EPE_RLNN[count] = (1#np.mean(discount_factors)
            #                               *np.mean(self.live_paths_array[tenor_idx])
            #                               *np.mean(self.direct_price_estimator))
            # print(f'mean exposure: {self.mean_EPE_RLNN[count]}')
            # print('directly evaluated mean exposure: '
            #       + f'{np.mean(discount_factors)*np.mean(self.live_paths_array[tenor_idx])*np.mean(self.direct_price_estimator)}')
            
            # print(f'discount_factors[:10]: {discount_factors[:10]}')
            # print(f'live_paths_array[:10]: {self.live_paths_array[tenor_idx][:10]}')
            # print(f'direct_price_estimator[:10]: {self.direct_price_estimator[:10]}')
            
            if compare:
                LSM_instance.price_Bermudan_swaption_LSM_Q(2, self.fixed_rate, 
                                            self.moneyness, self.notional, 
                                            self.swaption_type, 
                                            self.tenor_structure, eval_time, 
                                            self.units_basis_points, 
                                            'swap', False, False, 
                                            'power', self.EPE_r_t_paths, False, 
                                            self.EPE_x_t_paths)
                pathwise_LSM_prices = LSM_instance.pathwise_LSM_prices
                LSM_live_paths = LSM_instance.live_paths_array[0]
                
                LSM_instance.price_Bermudan_swaption_LSM_Q(2, self.fixed_rate, 
                                            self.moneyness, self.notional, 
                                            self.swaption_type, 
                                            self.tenor_structure, self.tenor_structure[tenor_idx+2], 
                                            self.units_basis_points, 
                                            'swap', False, False, 
                                            'power', self.EPE_r_t_paths, False, 
                                            self.EPE_x_t_paths)
                future_LSM_prices = LSM_instance.pathwise_LSM_prices
                disc_cont_LSM = (eval_discount_factors(self.model.n_annual_trading_days, 
                                                      self.EPE_r_t_paths, eval_time, 
                                                      self.tenor_structure[tenor_idx])
                                 *future_LSM_prices)
                current_ex = np.maximum(price_forward_start_swap(
                    self.model.a_param, self.fixed_rate, 
                    self.model.n_annual_trading_days, 
                    self.notional, 'swap', 
                    False, self.EPE_r_t_paths, self.model.sigma, 
                    self.swaption_type, self.tenor_structure[tenor_idx+1:], 
                    eval_time, self.model.time_0_f_curve, 
                    self.model.time_0_P_curve, self.units_basis_points, 
                    self.EPE_x_t_paths), 0)
                print(f'np.mean(current_ex): {np.mean(current_ex)}')
                print(f'np.mean(disc_cont_LSM): {np.mean(disc_cont_LSM)}')
                LSM_live_paths = np.zeros(n_paths)
                itm_idxs = np.where(current_ex>disc_cont_LSM)[0]
                LSM_live_paths[itm_idxs] = 1
                print(f'LSM_live_paths: {LSM_live_paths}')
                print(f'len(LSM_live_paths): {len(LSM_live_paths)}')
                
                self.EPE_LSM[count] = (discount_factors*LSM_live_paths
                                       *np.maximum(pathwise_LSM_prices, 0))
                self.mean_EPE_LSM[count] = np.mean(self.EPE_LSM[count])
            
        
    def plot_EPE_profile(self, 
                         plot: str,
                         save_plot: bool = True,
                         units_basis_points: bool = False
                        ):
        # Logical checks
        if plot.lower() == 'rlnn' and not hasattr(self, 'EPE_RLNN'):
            raise ValueError('RLNN EPE profile not yet evaluated.')
        if plot.lower() == 'compare' and not hasattr(self, 'EPE_RLNN'):
            raise ValueError('RLNN EPE profile not yet evaluated.')
        if plot.lower() == 'compare' and not hasattr(self, 'EPE_LSM'):
            raise ValueError('LSM EPE profile not yet evaluated.')
        
        ## Plot results
        fig, ax = plt.subplots()
        plt.suptitle('Expected Positive Exposure Profile of\n' 
                     + f'{self.tenor_structure_notation} ' 
                     + f'Bermudan {self.swaption_type.capitalize()} ' 
                     + 'Swaption\n' 
                     + f'{int(self.moneyness*100)}% Moneyness')
        
        # Adjust plotting if data is in absolute units but plotting 
        # must be in basis points of the notional
        if units_basis_points and not self.units_basis_points:
            adjust_units = 10000/self.notional
        else:
            adjust_units = 1
        if plot.lower() == 'rlnn':
            ax.plot(self.EPE_eval_times, np.mean(self.EPE_RLNN, axis=1)*adjust_units, 
                   label='Replicating\nPortfolio', 
                   linewidth=1.5)
            ax.fill_between(self.EPE_eval_times, 
                            (np.mean(self.EPE_RLNN, axis=1) 
                              + 1.96*st.sem(self.EPE_RLNN, axis=1))
                            *adjust_units, 
                            (np.mean(self.EPE_RLNN, axis=1) 
                              - 1.96*st.sem(self.EPE_RLNN, axis=1))
                            *adjust_units, 
                            alpha=.5)
        elif plot.lower() == 'compare':
            ax.plot(self.EPE_eval_times, np.mean(self.EPE_RLNN, axis=1)*adjust_units, 
                   label='Replicating\nPortfolio', color='C0', linestyle='solid')
            ax.plot(self.EPE_eval_times, np.mean(self.EPE_LSM, axis=1)*adjust_units, 
                   label='LSM', color='C1', linestyle='dashed')
            ax.fill_between(self.EPE_eval_times, 
                            (np.mean(self.EPE_RLNN, axis=1) 
                              + 1.96*st.sem(self.EPE_RLNN, axis=1))
                            *adjust_units, 
                            (np.mean(self.EPE_RLNN, axis=1)
                              - 1.96*st.sem(self.EPE_RLNN, axis=1))
                            *adjust_units, color='C0',
                            alpha=.5)
            ax.fill_between(self.EPE_eval_times, 
                            (np.mean(self.EPE_LSM, axis=1) 
                              + 1.96*st.sem(self.EPE_LSM, axis=1))
                            *adjust_units, 
                            (np.mean(self.EPE_LSM, axis=1) 
                              - 1.96*st.sem(self.EPE_LSM, axis=1))
                            *adjust_units, color='C1', 
                            alpha=.5)
            
        # Set axis labels
        ax.set_xlabel('Time $t$ (years)')
        if units_basis_points or self.units_basis_points:
            ax.set_ylabel('$EPE(t)$ (basis points of the notional)')
        else:
            ax.set_ylabel('$EPE(t)$ (absolute units)')
        
        plt.ylim(top=plt.yticks()[0][-1])
        ax.legend(loc='best')
        plt.tight_layout()
        
        if save_plot:
            plot_name = (f'{self.tenor_structure_notation} Bermudan ' 
                         + f'{self.swaption_type.capitalize()} Swaption EPE ' 
                         + f'{int(self.moneyness)*100} moneynesss plot')
            file_dir_and_name = str(self.model.figures_dir + 'Mean_of_' 
                                    + (plot_name.replace(' ','_'))  + '-' 
                        + datetime.now().strftime('%Y-%m-%d_%H%M%S'))
            plt.savefig(file_dir_and_name, bbox_inches='tight')
            print('\nPlot was saved to ' + file_dir_and_name + '.png')
        
        plt.show()
        
    def write_EPE_profile(self,
                          compare: bool,
                          mean: bool = True, 
                          units_basis_points: bool = False
                         ):
        if not self.units_basis_points:
            adjust_units = 10000/self.notional if units_basis_points else 1
            
        if mean is True:
            file_dir_and_name = str(self.model.data_dir + 'Mean_of_' 
                                    + self.tenor_structure_notation 
                                    + '_Bermudan_' + self.swaption_type 
                                    + '_swaption_' + 'RLNN_EPE_profiles')
            np.savetxt(file_dir_and_name + '.txt', 
                       np.array((self.EPE_eval_times, 
                                 adjust_units*np.mean(self.EPE_RLNN, axis=-1))), 
                       delimiter=', ', fmt='%.10g', 
                       header=str(f'{np.shape(self.EPE_RLNN)[-1]:,} Monte Carlo paths. ' 
                                  + 'First line: time (years); second line: mean EPE profile'))
            print('Mean RLNN EPE profike was saved to '
                  + f'{self.model.data_dir}\Mean_of_{file_dir_and_name}.')
            
            file_dir_and_name = str(self.model.data_dir + 'SEM_of_' 
                                    + self.tenor_structure_notation 
                                    + '_Bermudan_' + self.swaption_type 
                                    + '_swaption_' + 'RLNN_EPE_profiles')
            np.savetxt(file_dir_and_name + '.txt', 
                       np.array((self.EPE_eval_times, 
                                 st.sem(adjust_units*self.EPE_RLNN, axis=-1))), 
                       delimiter=', ', fmt='%.10g', 
                       header=str(f'{np.shape(self.EPE_RLNN)[-1]:,} Monte Carlo paths. ' 
                                  + 'First line: time (years); second line: SEM of EPE profile'))
            print('SEM of RLNN EPE profiles were saved to '
                  + f'{self.model.data_dir}\SEM_of_{file_dir_and_name}.')
            
            if compare:
                file_dir_and_name = str(self.model.data_dir + 'Mean_of_' 
                                        + self.tenor_structure_notation 
                                        + '_Bermudan_' + self.swaption_type 
                                        + '_swaption_' + 'LSM_EPE_profiles')
                np.savetxt(file_dir_and_name + '.txt', 
                           np.array((self.EPE_eval_times, 
                                     adjust_units*np.mean(self.EPE_LSM, axis=-1))), 
                           delimiter=', ', fmt='%.10g', 
                           header=str(f'{np.shape(self.EPE_LSM)[-1]:,} Monte Carlo paths. ' 
                                      + 'First line: time (years); second line: mean EPE profile'))
                print('Mean sensitivities were saved to '
                      + f'{self.model.data_dir}\Mean_of_{file_dir_and_name}.')
                
                file_dir_and_name = str(self.model.data_dir + 'SEM_of_' 
                                        + self.tenor_structure_notation 
                                        + '_Bermudan_' + self.swaption_type 
                                        + '_swaption_' + 'LSM_EPE_profiles')
                np.savetxt(file_dir_and_name + '.txt', 
                           np.array((self.EPE_eval_times, 
                                     st.sem(adjust_units*self.EPE_LSM, axis=-1))), 
                           delimiter=', ', fmt='%.10g', 
                           header=str(f'{np.shape(self.EPE_LSM)[-1]:,} Monte Carlo paths. ' 
                                      + 'First line: time (years); second line: SEM of EPE profile'))
                print('SEM of sensitivities were saved to '
                      + f'{self.model.data_dir}\SEM_of_{file_dir_and_name}.')
                
        elif mean.lower() == 'both':
            # Save RLNN profile
            file_dir_and_name = str(self.model.data_dir
                                    + self.tenor_structure_notation 
                                    + '_Bermudan_' + self.swaption_type 
                                    + '_swaption_' + 'RLNN_EPE_profile.parquet')
            write_Parquet_data(self.model.experiment_dir, file_dir_and_name, 
                           self.EPE_RLNN)
            print('RLNN EPE profiles were saved to '
                  + f'{self.model.data_dir}\{file_dir_and_name}.')
            
            # Save LSM profile
            file_dir_and_name = str(self.model.data_dir
                                    + self.tenor_structure_notation 
                                    + '_Bermudan_' + self.swaption_type 
                                    + '_swaption_' + 'LSM_EPE_profile.parquet')
            write_Parquet_data(self.model.experiment_dir, file_dir_and_name, 
                           self.EPE_LSM)
            print('LSM EPE profiles were saved to '
                  + f'{self.model.data_dir}\{file_dir_and_name}.')
            
            # Save mean and SEM of profiles
            file_dir_and_name = str(self.model.data_dir + 'Mean_of_' 
                                    + self.tenor_structure_notation 
                                    + '_Bermudan_' + self.swaption_type 
                                    + '_swaption_' + 'RLNN_EPE_profiles')
            np.savetxt(file_dir_and_name + '.txt', 
                       np.array((self.EPE_eval_times, 
                                 adjust_units*np.mean(self.EPE_RLNN, axis=-1))), 
                       delimiter=', ', fmt='%.10g', 
                       header=str(f'{np.shape(self.EPE_RLNN)[-1]:,} Monte Carlo paths. ' 
                                  + 'First line: time (years); second line: mean EPE profile'))
            print('Mean RLNN EPE profike was saved to '
                  + f'{self.model.data_dir}\Mean_of_{file_dir_and_name}.')
            
            file_dir_and_name = str(self.model.data_dir + 'SEM_of_' 
                                    + self.tenor_structure_notation 
                                    + '_Bermudan_' + self.swaption_type 
                                    + '_swaption_' + 'RLNN_EPE_profiles')
            np.savetxt(file_dir_and_name + '.txt', 
                       np.array((self.EPE_eval_times, 
                                 st.sem(adjust_units*self.EPE_RLNN, axis=-1))), 
                       delimiter=', ', fmt='%.10g', 
                       header=str(f'{np.shape(self.EPE_RLNN)[-1]:,} Monte Carlo paths. ' 
                                  + 'First line: time (years); second line: SEM of EPE profile'))
            print('SEM of RLNN EPE profiles were saved to '
                  + f'{self.model.data_dir}\SEM_of_{file_dir_and_name}.')
            
            if compare:
                file_dir_and_name = str(self.model.data_dir + 'Mean_of_' 
                                        + self.tenor_structure_notation 
                                        + '_Bermudan_' + self.swaption_type 
                                        + '_swaption_' + 'LSM_EPE_profiles')
                np.savetxt(file_dir_and_name + '.txt', 
                           np.array((self.EPE_eval_times, 
                                     adjust_units*np.mean(self.EPE_LSM, axis=-1))), 
                           delimiter=', ', fmt='%.10g', 
                           header=str(f'{np.shape(self.EPE_LSM)[-1]:,} Monte Carlo paths. ' 
                                      + 'First line: time (years); second line: mean EPE profile'))
                print('Mean sensitivities were saved to '
                      + f'{self.model.data_dir}\Mean_of_{file_dir_and_name}.')
                
                file_dir_and_name = str(self.model.data_dir + 'SEM_of_' 
                                        + self.tenor_structure_notation 
                                        + '_Bermudan_' + self.swaption_type 
                                        + '_swaption_' + 'LSM_EPE_profiles')
                np.savetxt(file_dir_and_name + '.txt', 
                           np.array((self.EPE_eval_times, 
                                     st.sem(adjust_units*self.EPE_LSM, axis=-1))), 
                           delimiter=', ', fmt='%.10g', 
                           header=str(f'{np.shape(self.EPE_LSM)[-1]:,} Monte Carlo paths. ' 
                                      + 'First line: time (years); second line: SEM of EPE profile'))
                print('SEM of sensitivities were saved to '
                      + f'{self.model.data_dir}\SEM_of_{file_dir_and_name}.')
        elif mean is False:
            # Save RLNN profiles
            file_dir_and_name = str(self.model.data_dir
                                    + self.tenor_structure_notation 
                                    + '_Bermudan_' + self.swaption_type 
                                    + '_swaption_' + 'RLNN_EPE_profile.parquet')
            write_Parquet_data(self.model.experiment_dir, file_dir_and_name, 
                           self.EPE_RLNN)
            print('RLNN EPE profiles were saved to '
                  + f'{self.model.data_dir}\{file_dir_and_name}.')
            
            if compare:
                # Save LSM profiles
                file_dir_and_name = str(self.model.data_dir
                                        + self.tenor_structure_notation 
                                        + '_Bermudan_' + self.swaption_type 
                                        + '_swaption_' + 'LSM_EPE_profile.parquet')
                write_Parquet_data(self.model.experiment_dir, file_dir_and_name, 
                               self.EPE_LSM)
                print('LSM EPE profiles were saved to '
                      + f'{self.model.data_dir}\{file_dir_and_name}.')
        
            
    def eval_forward_sensitivities(self, 
                            eval_times: float,
                            tenors: str,
                            bump_size: float = .0001,
                            plot: bool = False,
                            r_t_paths: list = None,
                            verbose: bool = False,
                            x_t_paths: list = None
                           ):
        """
        Info: Currenly implemented for time-zero Deltas only...
        """
        print(f'RLNN n_eval_times: {len(eval_times)}')
        ISDA_SIMM_tenors = np.array(list(self.model.ISDA_SIMM_tenors_dict.values()))
        self.bump_size = bump_size
        
        ## Input checks
        # Check whether short rate and zero-mean process paths were passed
        if r_t_paths is None and x_t_paths is not None:
            raise ValueError('pass corresponding short rate and zero-mean ' 
                             + 'process paths for determination of ' 
                             + 'pathwise stopping times')
        elif r_t_paths is not None and x_t_paths is None:
            raise ValueError('pass corresponding short rate and zero-mean ' 
                             + 'process paths for determination of ' 
                             + 'pathwise stopping times')
        elif r_t_paths is None and x_t_paths is None:
            r_t_paths = self.model.r_t_paths[:,self.x_test_idxs]
            x_t_paths = self.model.x_t_paths[:,self.x_test_idxs]
            n_paths = self.n_test
        else:
            n_paths = np.shape(x_t_paths)[1]
            
        # Check whether tenor time(s) passed as string or float(s)
        if type(tenors) is str:
            if tenors.lower() == 'isda-simm':
                # Select the relevant ISDA-SIMM tenors, i.e., those that occur 
                # during the swaption lifetime
                max_idx = np.searchsorted(ISDA_SIMM_tenors, self.maturity)
                self.Delta_tenors = ISDA_SIMM_tenors[:max_idx+1]
            else:
                raise ValueError('forward Delta sensitivity evaluation times ' 
                                 + 'not recognized. Enter as "ISDA-SIMM" to ' 
                                 + 'evaluate on all ISDA-SIMM tenors occuring ' 
                                 + 'during the swaption lifetime or enter as ' 
                                 + 'a float or list of floats with values ' 
                                 + 'equal to or smaller than last monitor date.')
        elif tenors is None:
            tenors = np.array([])
        elif not isinstance(tenors, np.ndarray):
            tenors = np.array(tenors)
            
            # Check whether last tenor time occurs after swaption expiry
            if tenors[-1] > self.maturity:
                raise ValueError('forward Delta sensitivity evaluation times ' 
                                 + 'may not exceed last monitor date.')
                
            # Add relevant ISDA-SIMM tenors to Delta tenors
            ISDA_SIMM_tenors = np.array(list(self.model.ISDA_SIMM_tenors_dict.values()))
            self.Delta_tenors = tenors
        n_Delta_tenors = len(self.Delta_tenors)
        
        # Evaluation times
        if not isinstance(eval_times, np.ndarray):
            eval_times = np.array([eval_times])
        if (eval_times>self.maturity).any():
            raise ValueError('forward sensitivity evaluation times may not occur ' 
                             + f'after last exercise date T={self.maturity}')
        self.Delta_eval_times = eval_times[eval_times<self.monitor_dates[-1]]
        
        # Remove evaluation times that coincide with any of the monitor dates 
        # as these times lead to tau_m and/or tau_M being equal to zero with 
        # which no interpolation of zero rates can be done
        remove_idxs = np.in1d(self.Delta_eval_times, self.monitor_dates)
        self.Delta_eval_times = self.Delta_eval_times[~remove_idxs]
        n_eval_times = len(self.Delta_eval_times)
        
        # Initialize the dVdR matrix
        self.dVdR = np.zeros((n_eval_times, n_Delta_tenors, n_paths))
        
        # Check whether exercise strategies have already been evaluated
        if not hasattr(self, 'pathwise_stopping_times'):
            self.eval_exercise_strategies(r_t_paths, x_t_paths)

        ## Compute forward deltas
        # Loop over evaluation times
        for eval_count, eval_time in enumerate(self.Delta_eval_times):
            if verbose:
                print(f'\n\nEvaluating forward dVdR at time {eval_time}...')
            
            # Select the portfolio that replicates the Bermudan swaption at 
            # the current evaluation time along with the strikes and weights
            (portfolio_date, __, 
             strikes, weights)      = self.select_portfolio(eval_time, n_paths)
            portfolio_count = np.searchsorted(self.monitor_dates, eval_time)
            
            t_idx = int(eval_time*self.model.n_annual_trading_days)
            P_t_m = price_zero_coupon_bond(self.model.a_param, 
                                            self.model.n_annual_trading_days, 
                                            r_t_paths[t_idx], self.model.sigma, 
                                            eval_time, portfolio_date, 
                                            self.model.time_0_f_curve, 
                                            self.model.time_0_P_curve, 
                                            x_t_paths[t_idx])
            P_t_M = price_zero_coupon_bond(self.model.a_param, 
                                            self.model.n_annual_trading_days, 
                                            r_t_paths[t_idx], self.model.sigma, 
                                            eval_time, self.maturity, 
                                            self.model.time_0_f_curve, 
                                            self.model.time_0_P_curve, 
                                            x_t_paths[t_idx])

            # Loop over tenors
            for tenor_count, tenor in enumerate(self.Delta_tenors):
                # print(f'\nEvaluating time {eval_time} dVdR_k for k={tenor}')
                
                # Initialize fractions and pathwise zero-coupon bond prices
                alpha_m = 0.
                alpha_M = 0.
                
                # Evaluate P_t_m
                tau_m = portfolio_date - eval_time
                
                if tau_m in ISDA_SIMM_tenors:
                    if tenor == tau_m:
                        alpha_m = 1.
                else:
                    left_bucket = ISDA_SIMM_tenors[ISDA_SIMM_tenors<tau_m][-1]
                    right_bucket = ISDA_SIMM_tenors[ISDA_SIMM_tenors>tau_m][0]
                    
                    if tenor == left_bucket or tenor == right_bucket:
                        alpha = (right_bucket - tau_m)/(right_bucket - left_bucket)
                        alpha_m = alpha if tenor == left_bucket else 1 - alpha
                
                # Evaluate P_t_M
                tau_M = self.maturity - eval_time
                # print(f'tau_M: {tau_M}')
                
                if tau_M in ISDA_SIMM_tenors:
                    if tenor == tau_M:
                        alpha_M = 1.
                        
                else:
                    left_bucket = ISDA_SIMM_tenors[ISDA_SIMM_tenors<tau_M][-1]
                    right_bucket = ISDA_SIMM_tenors[ISDA_SIMM_tenors>tau_M][0]
                    
                    if tenor == left_bucket or tenor == right_bucket:
                        alpha = (right_bucket - tau_M)/(right_bucket - left_bucket)
                        alpha_M = alpha if tenor == left_bucket else 1 - alpha
                    
                ## Evaluate the dVdR_k vector with k the current ISDA-SIMM 
                ## tenor
                
                dVdR_k = np.zeros((len(strikes), n_paths))
                
                for strike_count, strike in enumerate(strikes):
                    weight = weights[strike_count]
                    B_Tm_TM = B_func(self.model.a_param, portfolio_date, self.maturity)
                    sigma_p = self.model.sigma*np.sqrt((1 - np.exp(-2*self.model.a_param*(portfolio_date - eval_time)))
                                            /(2*self.model.a_param))*B_Tm_TM
                        
                    if 0. in P_t_m or 0. in P_t_M:
                        h = 0.
                    else:
                        h = 1/sigma_p*np.log(P_t_M/(P_t_m*strike)) + sigma_p/2
                    delta_payoff = 1 if self.swaption_type == 'payer' else -1
                    
                    dVdR_k[strike_count] = (bump_size*weight
                                            *(-1*delta_payoff*st.norm.cdf(delta_payoff*h)
                                              *alpha_M*tau_M*P_t_M 
                                              + delta_payoff*strike*
                                              st.norm.cdf(delta_payoff*(h - sigma_p))
                                              *alpha_m*tau_m*P_t_m))
                
                self.dVdR[eval_count,tenor_count] = (self.live_paths_array[portfolio_count]
                                                     *np.sum(dVdR_k, axis=0))
                
        # Adjust units to basis points of the notional
        if self.units_basis_points:
            self.dVdR = self.dVdR*10000/self.notional
            
    def write_sensitivities(self,
                            mean: bool = True, 
                            units_basis_points: bool = False
                           ):
        if not self.units_basis_points:
            adjust_units = 10000/self.notional if units_basis_points else 1
        
        for tenor_count, tenor in enumerate(self.Delta_tenors):
            label_tenor = f'{list(self.model.ISDA_SIMM_tenors_dict.keys())[list(self.model.ISDA_SIMM_tenors_dict.values()).index(tenor)]}'
            file_dir_and_name = str(self.model.data_dir
                                    + self.tenor_structure_notation 
                                    + '_Bermudan_' + self.swaption_type 
                                    + '_swaption_' + 'sensitivities_k=' 
                                    + label_tenor + '.parquet')
            if mean is True:
                file_dir_and_name = str(self.model.data_dir + 'Mean_of_' 
                                        + self.tenor_structure_notation 
                                        + '_Bermudan_' + self.swaption_type 
                                        + '_swaption_' + 'sensitivities_k=' 
                                        + label_tenor)# + '.parquet')
                np.savetxt(file_dir_and_name + '.txt', 
                           np.array((self.Delta_eval_times, 
                                     adjust_units*np.mean(self.dVdR[:,tenor_count,:], 
                                                          axis=-1))), 
                           delimiter=', ', fmt='%.10g', 
                           header=str(f'{np.shape(self.dVdR)[-1]:,} Monte Carlo paths. ' 
                                      + 'First line: time (years); second line: mean sensitivity'))
                print('Mean sensitivities were saved to '
                      + f'{self.model.data_dir}\Mean_of_{file_dir_and_name}.')
                
                file_dir_and_name = str(self.model.data_dir + 'SEM_of_' 
                                        + self.tenor_structure_notation 
                                        + '_Bermudan_' + self.swaption_type 
                                        + '_swaption_' + 'sensitivities_k=' 
                                        + label_tenor)# + '.parquet')
                np.savetxt(file_dir_and_name + '.txt', 
                           np.array((self.Delta_eval_times, 
                                     st.sem(adjust_units*self.dVdR[:,tenor_count,:], 
                                            axis=-1))), 
                           delimiter=', ', fmt='%.10g', 
                           header=str(f'{np.shape(self.dVdR)[-1]:,} Monte Carlo paths. ' 
                                      + 'First line: time (years); second line: SEM of sensitivity'))
                print('SEM of sensitivities were saved to '
                      + f'{self.model.data_dir}\SEM_of_{file_dir_and_name}.')
            elif mean.lower() == 'both':
                write_Parquet_data(self.model.experiment_dir, file_dir_and_name, 
                               self.dVdR[:,tenor_count,:])
                print('Sensitivities were saved to '
                      + f'{self.model.data_dir}\\{file_dir_and_name}.')
                
                file_dir_and_name = str(self.model.data_dir + 'Mean_of_' 
                                        + self.tenor_structure_notation 
                                        + '_Bermudan_' + self.swaption_type 
                                        + '_swaption_' + 'sensitivities_k=' 
                                        + label_tenor)# + '.parquet')
                np.savetxt(file_dir_and_name + '.txt', 
                           np.array((self.Delta_eval_times, 
                                     np.mean(adjust_units*self.dVdR[:,tenor_count,:], 
                                             axis=-1))), 
                           delimiter=', ', fmt='%.10g', 
                           header=str(f'{np.shape(self.dVdR)[-1]:,} Monte Carlo paths. ' 
                                      + 'First line: time (years); second line: mean sensitivity'))
                print('Mean sensitivities were saved to '
                      + f'{self.model.data_dir}\Mean_of_{file_dir_and_name}.')
                
                file_dir_and_name = str(self.model.data_dir + 'SEM_of_' 
                                        + self.tenor_structure_notation 
                                        + '_Bermudan_' + self.swaption_type 
                                        + '_swaption_' + 'sensitivities_k=' 
                                        + label_tenor)# + '.parquet')
                np.savetxt(file_dir_and_name + '.txt', 
                           np.array((self.Delta_eval_times, 
                                     st.sem(adjust_units*self.dVdR[:,tenor_count,:], 
                                            axis=-1))), 
                           delimiter=', ', fmt='%.10g', 
                           header=str(f'{np.shape(self.dVdR)[-1]:,} Monte Carlo paths. ' 
                                      + 'First line: time (years); second line: SEM of sensitivity'))
                print('SEM of sensitivities were saved to '
                      + f'{self.model.data_dir}\SEM_of_{file_dir_and_name}.')
            elif mean is False:
                write_Parquet_data(self.model.experiment_dir, file_dir_and_name, 
                               self.dVdR[:,tenor_count,:])
                print('Sensitivities were saved to '
                      + f'{self.model.data_dir}\{file_dir_and_name}.')
            
    def plot_forward_sensitivities(self, 
                                   plot: str,
                                   save_plot: bool = True,
                                   tenors: list = None,
                                   time_frame: list = None,
                                   units_basis_points: bool = False
                                  ):
        # Check whether forward Deltas were previously evaluated
        if not hasattr(self, 'dVdR'):
            raise ValueError('forward sensitivities have not yet been evaluated.')
            
        # If passed, get the indices of the passed time frame for plotting
        if isinstance(time_frame, list):
            if len(time_frame) == 2:
                time_frame_left_idx = np.searchsorted(self.Delta_eval_times, 
                                                      time_frame[0])
                time_frame_right_idx = np.searchsorted(self.Delta_eval_times, 
                                                       time_frame[1])
                if (time_frame[0] < 0 or time_frame[0] > time_frame[1] 
                    or time_frame[1] > self.monitor_dates[-1]):
                    raise ValueError('plotting time frame must be within the '
                                     + 'lifetime of the Bermudan swaption.')
            else:
                raise ValueError('plotting time frame must be passed as a ' 
                                 + 'list containing the start and end of the '
                                 + 'time frame.')
        else:
            time_frame_left_idx = 0
            time_frame_right_idx = -1
            
        # If passed, only plot the tenors specified by the user
        if isinstance(tenors, list):
            if all(tenor in self.Delta_tenors for tenor in tenors):
                tenor_idxs = np.searchsorted(self.Delta_tenors, tenors)#np.where(self.Delta_tenors==tenors)[0]
            else:
                raise ValueError('plotting tenors must be contained in the ' 
                                 + 'set of evaluated tenors '
                                 + f'{self.Delta_tenors}.')
        else:
            tenor_idxs = np.arange(0, len(self.Delta_tenors))
            
        ## Plot results
        if plot.lower() == 'rlnn':
            fig, ax = plt.subplots()
            plt.suptitle(r'$\frac{\partial \hat{{V}}}{\partial R_k}$ of ' 
                         + 'Replicating Portfolio of\n'
                         + f'{self.tenor_structure_notation} ' 
                         + f'Bermudan {self.swaption_type.capitalize()} ' 
                         + f'Swaption ({int(self.moneyness*100)}% Moneyness)')
            line_styles = ['dotted', 'solid', 'dashed']
            n_repeats = 4

            for count, tenor in enumerate(self.Delta_tenors[tenor_idxs]):
                tenor_count = tenor_idxs[count]
                if tenor in list(self.model.ISDA_SIMM_tenors_dict.values()):
                    # list(mydict.keys())[list(mydict.values()).index(16)]
                    label_tenor = f'{list(self.model.ISDA_SIMM_tenors_dict.keys())[list(self.model.ISDA_SIMM_tenors_dict.values()).index(tenor)]}'
                    label = label_tenor if tenor < 1. else label_tenor.upper()
                else:
                    label = str(tenor)
                label = '$k$='+label
                
                # Adjust plotting if data is in absolute units but plotting 
                # must be in basis points of the notional
                if units_basis_points and not self.units_basis_points:
                    ax.plot(self.Delta_eval_times[time_frame_left_idx:time_frame_right_idx], 
                            np.mean(self.dVdR[time_frame_left_idx:time_frame_right_idx,tenor_count,:], 
                                    axis=1)*10000/self.notional, 
                           label=label, linestyle=line_styles[tenor_count//n_repeats], 
                           linewidth=1.5)
                else:
                    ax.plot(self.Delta_eval_times[time_frame_left_idx:time_frame_right_idx], 
                            np.mean(self.dVdR[time_frame_left_idx:time_frame_right_idx,tenor_count,:], 
                                    axis=1), 
                           label=label, linestyle=line_styles[tenor_count//n_repeats], 
                           linewidth=1.5)
                    
            ax.set_xlabel('Time $t$ (years)')
            if units_basis_points or self.units_basis_points:
                ax.set_ylabel('Basis Points of the Notional')
                
            # Add legend (credit to Joe Kington at 
            # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot)
            # Shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            
            plt.ylim(top=plt.yticks()[0][-1])
            plt.ylim(bottom=plt.yticks()[0][0])
            
            # Put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            
            plt.tight_layout()
            
            if save_plot:
                plot_name = f'{self.tenor_structure_notation} Bermudan {self.swaption_type.capitalize()} Swaption dVdR plot'
                file_dir_and_name = str(self.model.figures_dir + 'Mean_of_' 
                                        + (plot_name.replace(' ','_'))  + '-' 
                            + datetime.now().strftime('%Y-%m-%d_%H%M%S'))
                plt.savefig(file_dir_and_name, bbox_inches='tight')
                print('\nPlot was saved to ' + file_dir_and_name + '.png')
            
            plt.show()
    
    
    def eval_exercise_strategies(self,
                                 r_t_paths: list = None,
                                 x_t_paths: list = None
                                ):
        """
        Info: this methode valuates the optimal pathwise stopping times of the 
            Bermudan swaption.
        """
        if r_t_paths is None and x_t_paths is not None:
            raise ValueError('pass corresponding short rate and zero-mean ' 
                             + 'process paths for determination of ' 
                             + 'pathwise stopping times')
        elif r_t_paths is not None and x_t_paths is None:
            raise ValueError('pass corresponding short rate and zero-mean ' 
                             + 'process paths for determination of ' 
                             + 'pathwise stopping times')
        elif r_t_paths is None and x_t_paths is None:
            eval_r_t_paths = self.model.r_t_paths[:,self.x_test_idxs]
            eval_x_t_paths = self.model.x_t_paths[:,self.x_test_idxs]
            n_paths = self.n_test
        else:
            # print(np.shape(x_t_paths))
            n_paths = np.shape(x_t_paths)[1]
            eval_r_t_paths = r_t_paths
            eval_x_t_paths = x_t_paths
            
        # Evaluate the replicating portfolio price on each monitor date before 
        # the last. The continuation value at the last monitor date is zero
        eval_times = self.monitor_dates
        n_eval_times = len(eval_times)
        
        # Construct pathwise stopping times matrix
        self.live_paths_array = np.ones((n_eval_times, n_paths), 
                                                  dtype=int)

        self.pathwise_stopping_times = np.zeros_like(self.live_paths_array, 
                                                 dtype=int)
        
        for count, eval_time in enumerate(eval_times[-1::-1]):
            rev_count = n_eval_times - count - 1
            print(f'rev_count: {rev_count}')
            print(f'eval_time: {eval_time}')
            print(f'eval_times[rev_count]: {eval_times[rev_count]}')
            print(f'self.tenor_structure[rev_count:]: {self.tenor_structure[rev_count:]}')
            
            ## Evaluate discounted continuation value at current evaluation 
            ## date
            if count == 0:
                disc_V_continuation = np.zeros(n_paths)
            else:
                self.price_direct_estimator(eval_times[rev_count+1], 
                                            eval_r_t_paths, eval_x_t_paths)
                discount_factors = eval_discount_factors(
                                            self.model.n_annual_trading_days, 
                                            eval_r_t_paths, eval_time, 
                                            eval_times[rev_count+1])
                disc_V_continuation = np.maximum(discount_factors
                                                 *self.direct_price_estimator, 
                                                 0)
            
            ## Evaluate exercise value at current evaluation date
            V_exercise = np.maximum(price_forward_start_swap(
                                    self.model.a_param, self.fixed_rate, 
                                    self.model.n_annual_trading_days, 
                                    self.notional, 'swap', False, None, 
                                    self.model.sigma, self.swaption_type, 
                                    self.tenor_structure[rev_count:], 
                                    eval_time, self.model.time_0_f_curve, 
                                    self.model.time_0_P_curve, 
                                    self.units_basis_points, eval_x_t_paths), 
                                    0)
            
            exercise_idxs = np.where(V_exercise>disc_V_continuation)[0]
            self.pathwise_stopping_times[rev_count,exercise_idxs] = 1
            self.pathwise_stopping_times[rev_count+1:,exercise_idxs] = 0
            # print(f'exercise_idxs: {exercise_idxs}')
            # print(f'Number of exercised paths: {len(exercise_idxs)}')
            self.live_paths_array[rev_count+1:,exercise_idxs] = 0
            
            # print(f'current live_paths_array: {self.live_paths_array[:,:10]}')
            
            # print(f'live_paths_array: {self.live_paths_array[:,:10]}')
            
        # print('number of paths with multiple stopping times: ' 
        #       + f'{np.nonzero(np.where(self.live_paths_array[0]+self.live_paths_array[1]+self.live_paths_array[2]+self.live_paths_array[3]+self.live_paths_array[4]>np.ones(n_paths),1,0))}')
        
        # print(f'Number of exercised paths: {len(self.live_paths_array[self.live_paths_array!=0.])}')
        
        # unexercised_idxs = np.where(self.live_paths_array[-1,:]>0)[0]
        # self.live_paths_array[:,unexercised_idxs] = 0
        
    
    def select_portfolio(self, 
                         eval_time: float,
                         n_paths: int
                        ):
        """
        Info: this method selects the replicating zero-coupon bond option 
            portfolio that corresponds to the specified evaluation time. 
            Specifically, it returns the portfolio weights, strikes, date, 
            and a zero-initialized numpy array for use in the evaluation of 
            the Bermudan swaption price.
        """
            
        for count, monitor_date in enumerate(self.monitor_dates):
            if eval_time <= monitor_date:
                portfolio_date = monitor_date
                strikes = self.ZBO_portfolio_strikes_list[count]
                weights = self.ZBO_portfolio_weights_list[count]
                replicating_portfolio = np.zeros((len(weights), n_paths))
                
                break
                
        return portfolio_date, replicating_portfolio, strikes, weights
    
    def visualize_neural_network(self, 
                                 file_name: str
                                ):
        visualize_neural_network(self.model.experiment_dir, file_name, 
                                 self.neural_network)
        
class EuropeanSwaption:
    """
    This class represents a European forward start interest swap option with 
    pricing methods for approximate and exact values. 
    
    Attributes: 
        fixed_rate: a float specifying the fixed interest rate used as the 
            strike rate of the (underlying) interest rate swap.
        
        model: an instance of the OneFactorHullWhiteModel class containing the 
            fundamental interest rate curves and simulated short rates.
            
        moneyness: a float specifying the level of moneyness e.g. 1.0 equals 
            100% moneyness.
            
        notional: a float specifying the notional amount of the (underlying) 
            interest rate swap.
            
        swaption_type: a str specifying the swaption type which can be 
            'payer' or 'receiver'.
            
        tenor_structure: a list containing the (underlying) forward swap's 
            starting date as the first entry and the payment dates as the 
            remaining entries.
            
        time_t: a float specifying the time of evaluation in years.
            
        units_basis_points: a bool which if True causes the output value to be 
            given in basis points of the notional.
            
    Methods:
        price_Bermudan_swaption_LSM_Q(): computes the LSM price of the Bermudan 
            swaption.
            
        price_European_swaption_MC_Q(): computes the Monte Carlo estimate of 
            the swaption price using the discounted pathwise payoffs.
            
        price_European_swaption_Bachelier(): computes the simplified Bachelier 
            model swaption price.
            
        price_European_swaption_exact_Jamshidian(): computes the exact 
            Jamshidian decomposition swaption price.
            
        eval_forward_sensitivities_bump_and_revalue_ISDA_SIMM(): computes the 
            forward sensitivities of the swaption using the bump-and-revalue 
            method according to the ISDA-SIMM specifications.
            
        eval_forward_sensitivities_Bachelier(): computes the simplified 
            Bachelier forward sensitivities of the swaption according to the 
            ISDA-SIMM specifications.
            
        eval_exposures_Bachelier(): computes the simplified Bachelier expected 
            positive exposure profile of the swaption.
        
    """
    
    def __init__(self,
                 fixed_rate: float,
                 model: object,
                 moneyness: float,
                 notional: float,
                 swaption_type: str,
                 tenor_structure: list,
                 time_t: float,
                 units_basis_points: bool
                ) -> object:
        self.fixed_rate = fixed_rate
        self.model = model
        self.notional = notional
        self.r_t_paths = self.model.r_t_paths
        self.swaption_type = swaption_type
        self.moneyness = moneyness
        self.tenor_structure = tenor_structure
        self.tenor = len(self.tenor_structure)
        self.time_t = time_t
        self.units_basis_points = units_basis_points
        self.x_t_paths = self.model.x_t_paths
        
        self.expiry = tenor_structure[0]
        self.maturity = tenor_structure[-1]
        
        self.tenor_structure_notation = (f'{round(self.tenor_structure[0])}Y' 
                                            + f'x{round(self.tenor_structure[-1] - self.tenor_structure[0])}Y')
        
        if moneyness:
            self.fixed_rate = eval_moneyness_adjusted_fixed_rate(moneyness, 
                                            self.model.n_annual_trading_days, 
                                            swaption_type, tenor_structure, 
                                            self.model.time_0_P_curve)
        
    def price_European_swaption_MC_Q(self,
                                     payoff_var: str,
                                     plot_timeline: str,
                                     verbose: bool = False,
                                    ):
        """
        Info: this method prices a European forward start interest rate swap 
            option within the one-factor Hull-White model under the 
            risk-neutral measure using Monte Carlo simulation.
            
        Input: 
            payoff_var: a str specifying the (underlying) swap payoff function: 
                if 'swap', the swap payoff is determined using the forward swap 
                rate; if 'forward', the swap payoff is determined using the 
                simply compounded forward rate.
                
            plot_timeline: a bool specifying whether or not the underlying swap 
                timeline is plotted and saved to the local folder.
                
            verbose: a bool which if False blocks the function prints.
            
        Output:
            self.pathwise_Monte_Carlo_prices: a 1D ndarray containing the 
                pathwise swaption prices.
                
            self.mean_Monte_Carlo_price: a float specifying the mean Monte 
                Carlo estimate of the swaption price.
                
            self.se_Monte_Carlo_price: a float specifying the standard error of 
                the mean Monte Carlo estimate of the swaption price.
        """
        # Check whether short rates were simulated
        if not hasattr(self.model, 'r_t_paths'):
            raise ValueError('no short rates have been simulated yet.')
            
        # Compute the Monte Carlo estimate of the swaption price
        self.payoff_var = payoff_var
        self.plot_timeline = plot_timeline
        self.pathwise_Monte_Carlo_prices = price_European_swaption_MC_Q(
                                                self.model.a_param, 
                                                self.model.experiment_dir, 
                                                self.fixed_rate, 
                                                self.model.n_annual_trading_days, 
                                                self.notional, self.payoff_var, 
                                                self.plot_timeline, 
                                                self.r_t_paths, 
                                                self.model.sigma, 
                                                self.swaption_type, 
                                                self.tenor_structure, 
                                                self.time_t, 
                                                self.model.time_0_f_curve, 
                                                self.model.time_0_P_curve, 
                                                self.units_basis_points, 
                                                self.x_t_paths, verbose)
        self.mean_Monte_Carlo_price = np.mean(self.pathwise_Monte_Carlo_prices)
        self.se_Monte_Carlo_price = st.sem(self.pathwise_Monte_Carlo_prices)
        
    def price_European_swaption_Bachelier(self,
                                          verbose: bool = False
                                         ):
        """
        Info: this method prices a European forward start interest rate swap 
            option within the one-factor Hull-White model  exactly using 
            Bachelier's option pricing framework.
            
        Input: 
            verbose: a bool which if False blocks the function prints.
            
        Output: 
            self.exact_price_Bachelier: a 1D ndarray containing the pathwise 
                simplified Bachelier model prices of the swaption.
                
            self.abs_err_Monte_Carlo_price: a 1D ndarray containing the 
                pathwise absolute price errors of the Monte Carlo estimates 
                with respect to the simplified Bachelier price.
                
            self.mean_abs_err_Monte_Carlo_price: a float specifying the mean 
                absolute price error of the Monte Carlo estimate with respect 
                to the simplified Bachelier price.
                
            self.se_abs_err_Monte_Carlo_price: a float specifying the standard 
                error of the mean absolute price error of the Monte Carlo 
                estimate with respect to the simplified Bachelier price.
        """
        self.exact_price_Bachelier = price_European_swaption_exact_Bachelier(
                                                self.model.a_param, self.fixed_rate, 
                                                self.moneyness, 
                                                self.model.n_annual_trading_days, 
                                                self.notional, self.r_t_paths, 
                                                self.model.sigma, 
                                                self.swaption_type, 
                                                self.tenor_structure, self.time_t, 
                                                self.model.time_0_f_curve, 
                                                self.model.time_0_P_curve, 
                                                self.model.time_0_rate, 
                                                self.units_basis_points, 
                                                self.x_t_paths)
        
        if hasattr(self, 'mean_Monte_Carlo_price'):
            self.abs_err_Monte_Carlo_price = (self.mean_Monte_Carlo_price 
                                              - self.exact_price_Bachelier)
            self.mean_abs_err_Monte_Carlo_price = np.mean(self.abs_err_Monte_Carlo_price)
            self.se_abs_err_Monte_Carlo_price = st.sem(self.abs_err_Monte_Carlo_price)
        
    def price_European_swaption_exact_Jamshidian(self,
                                                 verbose: bool = False
                                                ):
        """
        Info: this method values a European forward start interest rate swap 
            option within the one-factor Hull-White model exactly using 
            Jamshidian decomposition.
            
        Input: 
            verbose: a bool which if False blocks the function prints.
            
        Output:
            self.exact_price_Jamshidian: a float specifying the exact price of 
                the swaption obtained using Jamshidian decomposition.
                
            self.abs_err_Monte_Carlo_price: a 1D ndarray containing the 
                pathwise absolute price errors of the Monte Carlo estimates 
                with respect to the exact Jamshidian price.
                
            self.mean_abs_err_Monte_Carlo_price: a float specifying the mean 
                absolute price error of the Monte Carlo estimate with respect 
                to the exact Jamshidian price.
                
            self.se_abs_err_Monte_Carlo_price: a float specifying the standard 
                error of the mean absolute price error of the Monte Carlo 
                estimate with respect to the exact Jamshidian price.
        """
        self.exact_price_Jamshidian = price_European_swaption_exact_Jamshidian(
                                            self.model.a_param, self.fixed_rate, 
                                            self.moneyness, 
                                            self.model.n_annual_trading_days, 
                                            self.notional, self.model.sigma, 
                                            self.swaption_type, 
                                            self.tenor_structure, self.time_t, 
                                            self.model.time_0_f_curve, 
                                            self.model.time_0_P_curve, 
                                            self.model.time_0_rate, 
                                            self.units_basis_points)
        
        if hasattr(self, 'mean_Monte_Carlo_price'):
            self.abs_err_Monte_Carlo_price = (self.mean_Monte_Carlo_price 
                                              - self.exact_price_Jamshidian)
            self.mean_abs_err_Monte_Carlo_price = np.mean(self.abs_err_Monte_Carlo_price)
            self.se_abs_err_Monte_Carlo_price = st.sem(self.abs_err_Monte_Carlo_price)
        
    def eval_forward_sensitivities_bump_and_revalue_ISDA_SIMM(self, 
                                                              bump_size: float, 
                                                              bump_time: float, 
                                                              eval_time: float, 
                                                              plot_bumped_curve: bool, 
                                                              bump_model: object = None,
                                                              seed: int = None,
                                                              verbose: bool = False
                                                             ):
        """
        Info: this method bumps the yield curve at one of the ISDA-SIMM risk 
            tenors and revalues the swaption price for the resulting bumped 
            short rates, then determines the sensitivity as the finite 
            difference of the bumped and unbumped swaption prices.
            
        Input:
            bump_size: the size of the bump for use in the resimulation of a 
                bumped set of short rates in the bump-and-revalue sensitivity 
                estimation.
                
            bump_time: the time of the zero rate curve bump for use in the
                resimulation of a bumped set of short rates in the 
                bump-and-revalue sensitivity estimation method. Can be entered 
                in years as type float or as a string specifying one of the 
                ISDA-SIMM tenors {2w, 1m, 3m, 6m, 1Y, 2Y, 3Y, 5Y, 10Y, 15Y, 
                                  20Y, 30Y} formatted as "ISDA-SIMM {tenor}".
                
            eval_time: a float specifying the time of evaluation in years.
            
            plot_bumped_curve: whether or not to plot the bumped fundamental 
                interest rate curves.
                
            bump_model: an instance of the OneFactorHullWhite class that 
                contains already-bumped interest rate curves and short rates.
                
            seed: an int specifying the seed for the randon number generator 
                used for the generation of the short rate paths.
        
        Output:
            self.pathwise_forward_sensitivities: a 1D ndarray containing the 
                pathwise forward sensitivities.
                
            self.mean_forward_sensitivity: a float specifying the mean forward 
                sensitivity.
                
            self.se_forward_sensitivities: a float specifying the standard 
                error of the mean forward sensitivity.
        """
        self.bump_size = bump_size
        self.bump_time = bump_time
        self.plot_bumped_curve = plot_bumped_curve
        self.seed = seed
        
        # Check whether price was already evaluated
        if (not hasattr(self, 'mean_Monte_Carlo_price') 
            or self.time_t != eval_time):
            print('Monte Carlo price not yet evaluated. Now pricing swaption...')
            self.price_European_swaption_MC_Q('swap', False)
        
        if type(bump_time) is str and bump_time.lower()[:9] != 'isda-simm':
            raise ValueError('bump time was not passed as one of the ISDA-' 
                             + 'SIMM tenors. Enter as "ISDA-SIMM" followed by ' 
                             + 'a space and one of the following:\n' 
                             + f'{self.model.ISDA_SIMM_tenors_dict.keys()}')
        
        bump_time_tenor = bump_time.split()[1].lower()
        if bump_time_tenor not in self.model.ISDA_SIMM_tenors_dict.keys():
            raise ValueError('bump time was not passed as one of the ISDA-' 
                             + 'SIMM tenors. Enter as a string starting with ' 
                             + '"ISDA-SIMM" followed by a space and one of the' 
                             + ' following keys:\n\t' 
                             + f'{list(self.model.ISDA_SIMM_tenors_dict.keys())}')
        bump_time = self.model.ISDA_SIMM_tenors_dict[bump_time_tenor]
        
        # Check whether a bumped 1F-HW model was passed or whether one should 
        # be created
        if bump_model is None:
            # Make a deep copy of the 1F-HW instance for bumping without 
            # altering the original instance's properties
            self.bump_model = copy.deepcopy(self.model)
            
            # Check whether previous model short rates were simulated using 
            # user-specified RNG seed
            if self.bump_model.seed is None and self.seed is not None:
                if verbose:
                    print('the original short rates were not simulated ' 
                          + 'with a user-specified RNG seed. Now simulating ' 
                          + f'passed model\'s short rates with seed {self.seed}.')
                    
                self.bump_model.seed = self.seed
                self.bump_model.sim_short_rate_paths(self.bump_model.a_param, 
                                self.bump_model.n_paths,
                                self.bump_model.r_t_sim_time, 
                                self.bump_model.sigma, 
                                self.bump_model.antithetic, self.seed, 
                                self.bump_model.r_t_process_type, 
                                self.bump_model.sim_type, verbose)
                
            elif (self.bump_model.seed is not None 
                  and self.bump_model.seed != self.seed):
                if verbose:
                    print('the specified RNG seed does not match the seed of ' 
                          + 'the original short rates. Now using the ' 
                          + f'passed model\'s seed {bump_model.seed}.')
                self.seed = self.bump_model.seed
            
            
            # Bump the yield curve
            self.bump_model.bump_yield_curve(self.bump_size, self.bump_time, 
                                              self.plot_bumped_curve)
            self.bump_model.resim_short_rates(self.bump_model.time_0_f_curve_bumped, 
                                    self.bump_model.time_0_dfdt_curve_bumped, 
                                    self.bump_model.seed)
            
        elif (hasattr(bump_model, 'time_0_R_curve_bumped') 
              and bump_model.seed == self.seed 
              and bump_model.n_paths == self.model.n_paths):
            self.bump_model = bump_model
            
        else:
            raise ValueError('Passed bumped model seed does not correspond.')
        
        # Revalue the swaption
        self.bump_model.time_0_R_curve = self.bump_model.time_0_R_curve_bumped
        self.bump_model.time_0_f_curve = self.bump_model.time_0_f_curve_bumped
        self.bump_model.time_0_dfdt_curve = self.bump_model.time_0_dfdt_curve_bumped
        self.bump_model.time_0_P_curve = self.bump_model.time_0_P_curve_bumped
        self.bump_model.r_t_paths = self.bump_model.r_t_paths_resim
        self.bump_model.x_t_paths = self.bump_model.x_t_paths_resim
        self.revalue_inst = EuropeanSwaption(self.fixed_rate, self.bump_model, 
                                             False, self.notional, 
                                             self.swaption_type, 
                                             self.tenor_structure, eval_time, 
                                             self.units_basis_points)
        self.revalue_inst.price_European_swaption_MC_Q('swap', False)
        self.pathwise_prices_reval = self.revalue_inst.mean_Monte_Carlo_price
        self.mean_price_reval = self.revalue_inst.mean_Monte_Carlo_price
        self.se_price_reval = self.revalue_inst.se_Monte_Carlo_price
        
        # Evaluate the pathwise sensitivities
        self.pathwise_forward_sensitivities = ((self.pathwise_prices_reval 
                                        - self.mean_Monte_Carlo_price))
        self.mean_forward_sensitivity = np.mean(self.pathwise_forward_sensitivities)
        self.se_forward_sensitivities = st.sem(self.pathwise_forward_sensitivities)
        
    def eval_forward_sensitivities_Bachelier(self,
                                             bump_size: float,
                                             eval_times: list,
                                             tenors: list,
                                             r_t_paths: list = None,
                                             x_t_paths: list = None,
                                             verbose: bool = False
                                            ) -> float:
        """
        Info: this method computes the simplified Bachelier forward 
            sensitivities of the swaption according to the ISDA-SIMM 
            specifications.
            
        Input: 
            bump_size: the size of the bump for use in the resimulation of a 
                bumped set of short rates in the bump-and-revalue sensitivity 
                estimation.
                
            eval_times: a list containing the sensitivity evaluation times.
            
            tenors: a list containing the risk tenors with respect to to which 
                the sensitivities are computed.
                
            r_t_paths: a 2D ndarray containing the simulated short rate paths 
                along a number of columns corresponding to the number of paths 
                and a number of rows being a discrete short rate time series of 
                length equal to the total number of trading days.
                
            x_t_paths: a 2D ndarray containing the zero-mean process paths 
                along a number of columns corresponding to the number of paths 
                and a number of rows being a discrete short rate time series of 
                length equal to the total number of trading days.
                
            verbose: a bool which if False blocks the function prints.
            
        Output: 
            self.dVdR_Bachelier: a 3D array containing the pathwise simplified
                Bachelier sensitivities computed at the evaluation times along 
                axis 0 for the risk tenors along axis 1.
        """
        # Determine payoff type
        if self.swaption_type == 'payer':
            delta_payoff = 1
        elif self.swaption_type == 'receiver':
            delta_payoff = -1
            
        ## Input checks
        # Check whether short rate and zero-mean process paths were passed
        if r_t_paths is None and x_t_paths is not None:
            raise ValueError('pass corresponding short rate and zero-mean ' 
                             + 'process paths for determination of ' 
                             + 'pathwise stopping times')
        elif r_t_paths is not None and x_t_paths is None:
            raise ValueError('pass corresponding short rate and zero-mean ' 
                             + 'process paths for determination of ' 
                             + 'pathwise stopping times')
        elif r_t_paths is None and x_t_paths is None:
            self.sensitivities_r_t_paths = self.model.r_t_paths
            self.sensitivities_x_t_paths = self.model.x_t_paths
            n_paths = self.model.n_paths
        else:
            self.sensitivities_r_t_paths = r_t_paths
            self.sensitivities_x_t_paths = x_t_paths
            n_paths = np.shape(x_t_paths)[1]
            
        # ISDA SIMM tenors
        ISDA_SIMM_tenors = np.array(list(self.model.ISDA_SIMM_tenors_dict.values()))
            
        # Check whether tenor time(s) passed as string or float(s)
        if type(tenors) is str:
            if tenors.lower() == 'isda-simm':
                # Select the relevant ISDA-SIMM tenors, i.e., those that occur 
                # during the swaption lifetime
                max_idx = np.searchsorted(ISDA_SIMM_tenors, self.maturity)
                self.Delta_tenors_analytic = ISDA_SIMM_tenors[:max_idx+1]
            else:
                raise ValueError('forward Delta sensitivity evaluation times ' 
                                 + 'not recognized. Enter as "ISDA-SIMM" to ' 
                                 + 'evaluate on all ISDA-SIMM tenors occuring ' 
                                 + 'during the swaption lifetime or enter as ' 
                                 + 'a float or list of floats with values ' 
                                 + 'equal to or smaller than last monitor date.')
        elif tenors is None:
            tenors = np.array([])
        elif not isinstance(tenors, np.ndarray):
            tenors = np.array(tenors)
            
            # Check whether last tenor time occurs after swaption expiry
            if tenors[-1] > self.maturity:
                raise ValueError('forward Delta sensitivity evaluation times ' 
                                 + f'may not exceed swaption maturity T={self.maturity}')
                
            # Add relevant ISDA-SIMM tenors to Delta tenors
            ISDA_SIMM_tenors = np.array(list(self.model.ISDA_SIMM_tenors_dict.values()))
            self.Delta_tenors_analytic = tenors
        n_Delta_tenors = len(self.Delta_tenors_analytic)
        
        # Evaluation times
        if not isinstance(eval_times, np.ndarray):
            eval_times = np.array([eval_times])
        if (eval_times>self.expiry).any():
            raise ValueError('forward Delta evaluation times may not occur ' 
                             + f'after swaption expiry {self.expiry}')
        self.Delta_eval_times_analytic = eval_times[eval_times<self.expiry]
        n_eval_times = len(self.Delta_eval_times_analytic)
        
        # Initialize the dVdR matrix
        self.dVdR_Bachelier = np.zeros((n_eval_times, n_Delta_tenors, n_paths))
            
        ## Compute forward deltas
        # Loop over evaluation times
        for eval_count, eval_time in enumerate(self.Delta_eval_times_analytic):
            time_t_idx = int(eval_time*self.model.n_annual_trading_days)

            # # Evaluate annuity etc.
            ZCBs_curve_t = construct_zero_coupon_bonds_curve(self.model.a_param, 
                                        None, self.model.n_annual_trading_days, 
                                        False, r_t_paths, self.model.sigma, 
                                        self.maturity, eval_time, 
                                        self.model.time_0_f_curve, 
                                        self.model.time_0_P_curve, x_t_paths)
            swap_rate_t = eval_swap_rate(ZCBs_curve_t, 
                                        self.model.n_annual_trading_days, 
                                        self.swaption_type, 
                                        self.tenor_structure, eval_time)
            approx_vol = eval_approx_vol(self.model.a_param, 
                                         self.model.n_annual_trading_days, 
                                         self.model.sigma, self.swaption_type, 
                                         self.tenor_structure, eval_time, 
                                         self.model.time_0_f_curve, 
                                         self.model.time_0_rate, 
                                         self.model.time_0_P_curve)
            
            # Compute zero-coupon bond prices
            P_t_m = price_zero_coupon_bond(self.model.a_param, 
                                            self.model.n_annual_trading_days, 
                                            self.sensitivities_r_t_paths[time_t_idx], self.model.sigma, 
                                            eval_time, self.expiry, 
                                            self.model.time_0_f_curve, 
                                            self.model.time_0_P_curve, 
                                            self.sensitivities_x_t_paths[time_t_idx])
                    
            P_t_M = price_zero_coupon_bond(self.model.a_param, 
                                            self.model.n_annual_trading_days, 
                                            self.sensitivities_r_t_paths[time_t_idx], self.model.sigma, 
                                            eval_time, self.maturity, 
                                            self.model.time_0_f_curve, 
                                            self.model.time_0_P_curve, 
                                            self.sensitivities_x_t_paths[time_t_idx])
            
            # Loop over tenors
            for tenor_count, tenor in enumerate(self.Delta_tenors_analytic):
                
                # Evaluate sum term
                sum_term = np.zeros((self.tenor-1, n_paths))
                for date_count, date in enumerate(self.tenor_structure[1:]):
                    alpha_date = 0.
                    tau_date = date - eval_time # self.tenor_structure[date_count]
                    P_t_date = price_zero_coupon_bond(self.model.a_param,
                                                      self.model.n_annual_trading_days, 
                                                      self.sensitivities_r_t_paths[time_t_idx], self.model.sigma, 
                                                      eval_time, date, 
                                                      self.model.time_0_f_curve, 
                                                      self.model.time_0_P_curve, 
                                                      self.sensitivities_x_t_paths[time_t_idx])
                    
                    if tau_date in ISDA_SIMM_tenors:
                        if tenor == tau_date:
                            alpha_date = 1.
                    else:
                        left_bucket = ISDA_SIMM_tenors[ISDA_SIMM_tenors<tau_date][-1]
                        right_bucket = ISDA_SIMM_tenors[ISDA_SIMM_tenors>tau_date][0]
                        
                        if tenor == left_bucket or tenor == right_bucket:
                            alpha = (right_bucket - tau_date)/(right_bucket - left_bucket)
                            alpha_date = alpha if tenor == left_bucket else 1 - alpha
                    
                    sum_term[date_count] = alpha_date*tau_date*P_t_date
                    
                sum_term = np.sum(sum_term, axis=0)
                    
                
                # Evaluate fractions of expiry and maturity ZCB prices
                alpha_m = 0.
                alpha_M = 0.
                tau_M = self.maturity - eval_time
                tau_m = self.expiry - eval_time
                
                if tau_m in ISDA_SIMM_tenors:
                    if tenor == tau_m:
                        alpha_m = 1.
                else:
                    left_bucket = ISDA_SIMM_tenors[ISDA_SIMM_tenors<tau_m][-1]
                    right_bucket = ISDA_SIMM_tenors[ISDA_SIMM_tenors>tau_m][0]
                    
                    if tenor == left_bucket or tenor == right_bucket:
                        alpha = (right_bucket - tau_m)/(right_bucket - left_bucket)
                        alpha_m = alpha if tenor == left_bucket else 1 - alpha
                        
                if tau_M in ISDA_SIMM_tenors:
                    if tenor == tau_M:
                        alpha_M = 1.
                else:
                    left_bucket = ISDA_SIMM_tenors[ISDA_SIMM_tenors<tau_M][-1]
                    right_bucket = ISDA_SIMM_tenors[ISDA_SIMM_tenors>tau_M][0]
                    
                    if tenor == left_bucket or tenor == right_bucket:
                        alpha = (right_bucket - tau_M)/(right_bucket - left_bucket)
                        alpha_M = alpha if tenor == left_bucket else 1 - alpha
                        
            
                dVdR_k = (delta_payoff*sum_term
                          *(self.fixed_rate*st.norm.cdf(delta_payoff*(swap_rate_t 
                                                         - self.fixed_rate)
                                                        /approx_vol) 
                            - approx_vol*st.norm.pdf(delta_payoff*(self.fixed_rate 
                                                            - swap_rate_t)
                                                           /approx_vol)) 
                          + delta_payoff*st.norm.cdf(delta_payoff*(swap_rate_t - self.fixed_rate)
                                        /approx_vol)
                          *(alpha_M*tau_M*P_t_M - alpha_m*tau_m*P_t_m))
                
                self.dVdR_Bachelier[eval_count,tenor_count] = bump_size*dVdR_k
                
    def eval_exposures_Bachelier(self,
                                 plot_exposure: bool,
                                 time_t: float,
                                 eval_times: list = None,
                                 eval_time_spacing: str = 'monthly',
                                 units_basis_points: bool = False,
                                 verbose: bool = False,
                                ):
        """
        Info: this method evaluates the simplified Bachelier expected positive 
            exposure profile of a European swaption with daily, weekly, or 
            monthly evaluations, and plots the exposure profile if desired.
            
        Input:
            plot_exposure: a boolean specifying whether or not the exposures 
                are to be plotted.
                
            time_t: the time at which the exposures are evaluated.
            
            eval_times: the evaluation times of the exposure profile.
            
            eval_time_spacing: the time spacing of the exposure evaluations. 
                Can be set to daily, weekly, or monthly.
                
            units_basis_points: a bool which if True causes the output value to be 
            given in basis points of the notional.
                
            verbose: a bool which if False blocks the function prints.
        """
        self.time_t_EPE = time_t
        self.eval_time_spacing = eval_time_spacing
        n_paths = np.shape(self.model.x_t_paths)[1]
        
        if eval_times is not None:
            self.EPE_eval_times = eval_times
            self.n_eval_times = len(self.EPE_eval_times)
        else:
            if self.eval_time_spacing.lower() == 'daily':
                self.n_eval_times = int(self.model.n_annual_trading_days*self.expiry)
            elif self.eval_time_spacing.lower() == 'weekly':
                self.n_eval_times = int(52*self.expiry)
            elif self.eval_time_spacing.lower() == 'monthly':
                self.n_eval_times = int(12*self.expiry)
            self.EPE_eval_times = np.linspace(time_t, self.tenor_structure[0], 
                                              self.n_eval_times, endpoint=False)
            
        ## Evaluate swaption exposures. Due to the positive part operator in 
        ## the swaption payoff function, the exposures are simply equal to the 
        ## swaption prices at the evaluation dates discounted back to time_t.
        self.EPE_array = np.zeros((self.n_eval_times, n_paths))
        
        for count, eval_t in enumerate(self.EPE_eval_times):
            
            # eval_t_idx = int(eval_t*self.model.n_annual_trading_days)
            if units_basis_points and not self.units_basis_points:
                pricing_instance = EuropeanSwaption(self.fixed_rate, 
                                                    self.model, self.moneyness, 
                                                    self.notional, 
                                                    self.swaption_type, 
                                                    self.tenor_structure,
                                                    eval_t, units_basis_points)
            else:
                pricing_instance = EuropeanSwaption(self.fixed_rate, 
                                                    self.model, self.moneyness, 
                                                    self.notional, 
                                                    self.swaption_type, 
                                                    self.tenor_structure, 
                                                    eval_t, self.units_basis_points)
            pricing_instance.price_European_swaption_Bachelier()
            self.EPE_array[count] = np.maximum(pricing_instance.exact_price_Bachelier, 
                                               0)
            
        # Plot the EPE profile if specified
        if plot_exposure:
            plot_title = str('Expected Positive Exposure Profile of\n' 
                         + f'{self.tenor_structure_notation} ' 
                         + f'European {self.swaption_type.capitalize()} ' 
                         + 'Swaption\n' 
                         + f'{int(self.moneyness*100)}% Moneyness')
            y_label = '$EPE(t)$ '
            
            if self.units_basis_points:
                y_label = y_label + '(basis points of the notional)'
            elif units_basis_points and not self.units_basis_points:
                y_label = y_label + '(basis points of the notional)'
            plot_time_series(self.model.experiment_dir, self.model.n_annual_trading_days, 
                             True, plot_title, self.EPE_array)
        
class EuropeanSwaptionRLNN:
    """
    This class represents a European forward start interest rate swap option 
    under the one-factor Hull-White (1F-HW) short rate model as approximated by 
    the regress-later neural network (RLNN) method. 
    
    Info: the European swaption is valued within the 1F-HW model under the 
        risk-neutral measure Q using the RLNN method. A portfolio of 
        zero-coupon bond options or forwards is used for the semi-static 
        replication of the European swaption.
        
    Attributes:
        fixed_rate: a float specifying the fixed interest rate used as the 
            strike rate of the (underlying) interest rate swap.
        
        model: the one-factor Hull-White model instance containing the main 
            model parameters, fundamental interest rate curves, and simulated 
            short rate paths.
            
        notional: the notional of the European swaption.
        
        swaption_type: the European swaption type which can be 'payer' or 
            'receiver'.
            
        tenor_structure: a list containing the underlying forward swap's tenor 
            structure i.e. the starting date as the first entry and the payment 
            dates as the remaining entries.
            
        time_t: the time of valuation of the European swaption.
            
        input_dim: the dimensions of the input layer.
    
    Methods:
        replicate(): 

        train(): 
    
        construct_ZBO_portfolio(): 
    
        price_direct_estimator(): 
    
        eval_EPE_profile(): 
    
        write_EPE_profile(): 
    
        plot_EPE_profile(): 
    
        eval_forward_sensitivities(): 

    """
    
    def __init__(self,
                 fixed_rate: float,
                 model: object,
                 notional: float,
                 moneyness: bool,
                 swaption_type: str,
                 tenor_structure: list,
                 n_run = None,
                 units_basis_points: bool = False
                ) -> object:
        # Assign main parameters from constructor input
        self.fixed_rate = fixed_rate
        self.model = model
        self.notional  = notional
        self.moneyness = moneyness
        self.swaption_type = swaption_type
        self.tenor_structure = (tenor_structure if isinstance(tenor_structure, 
                                                              np.ndarray)
                                else np.array(tenor_structure))
        self.n_run = n_run
        self.units_basis_points = units_basis_points
        
        # Assign secondary parameters
        self.expiry = self.tenor_structure[0]
        self.monitor_dates = self.tenor_structure[:-1]
        self.maturity = self.tenor_structure[-1]
        self.tenor = len(self.tenor_structure)
        self.index = 0
        self.tenor_structure_notation = (f'{round(self.tenor_structure[0])}Y' 
                                            + f'x{round(self.tenor_structure[-1] - self.tenor_structure[0])}Y')
        
        # Adjust the fixed rate if a level of moneyness was specified
        if moneyness is not None:
            self.fixed_rate = eval_moneyness_adjusted_fixed_rate(moneyness, 
                                            self.model.n_annual_trading_days, 
                                            swaption_type, tenor_structure, 
                                            self.model.time_0_P_curve)
        
        # Initialize name of directory for storing neural network weights
        self.weights_dir = (self.model.data_dir 
                            + f'\\NN_weights_{self.tenor_structure[0]}Y' 
                            + f'x{self.tenor_structure[-1] - self.tenor_structure[0]}Y' 
                            + f'_N_={self.model.n_paths}' 
                            + f'_moneyness={self.moneyness}')
        
    def replicate(self,
                  neural_network: object,
                  time_t: float,
                  batch_size: int = 32,
                  input_dim: int = 1,
                  learn_rate: float = .0003,
                  n_epochs: int = 4500,
                  n_hidden_nodes: int = 64,
                  save_weights: bool = False,
                  seed_biases: int = None,
                  seed_weights: int = None,
                  test_fit: bool = False,
                  train_size: float = .2,
                 ):
        """
        Info: this method calls the training and portfolio construction methods 
            in order to replicate the Bermudan swaption value over its tenor 
            structure.
        """
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.learn_rate = learn_rate
        self.n_epochs = n_epochs
        self.n_hidden_nodes = n_hidden_nodes
        self.save_weights = save_weights
        self.seed_biases = seed_biases
        self.seed_weights = seed_weights
        self.test_fit = test_fit
        self.time_t = time_t
        self.train_size = train_size
        
        # Update weights directory name and create it
        self.weights_dir = (self.weights_dir 
                            + f'_learn_rate={np.format_float_positional(self.learn_rate, trim="-")}'
                            + f'_n_epochs={self.n_epochs}'
                            + f'_n_hidden_nodes={self.n_hidden_nodes}'
                            + f'_train_size={self.n_hidden_nodes}')
        if not os.path.isdir(self.weights_dir):
            os.makedirs(self.weights_dir)
        
        self.ZBO_portfolios_weights = np.zeros(self.n_hidden_nodes)
        self.ZBO_portfolios_strikes = np.zeros_like(self.ZBO_portfolios_weights)
        
        ## Replication of the European swaption using zero-coupon bond options
        self.t_maturity_idx = int(self.expiry*self.model.n_annual_trading_days)
        x_t_in = self.model.x_t_paths[self.t_maturity_idx]
        
        self.train(None, x_t_in)
        self.construct_ZBO_portfolio()
        
    def train(self,
              neural_network: object,
              x_t_input: list,
             ):
        """
        Info: this method prepares the data for training and initializes and 
            fits the neural network.
        """
        ## Split input zero-mean process data into training and test sets and 
        ## evaluate the current exercise value
        # Split the data and corresponding Monte Carlo path indices into 
        # training and test sets
        indices = np.arange(self.model.n_paths)
        (self.x_train, self.x_test, 
         self.x_train_idxs, self.x_test_idxs) = train_test_split(x_t_input, 
                                                    indices, 
                                                    test_size=1-self.train_size, 
                                                    random_state=False)
        self.n_train = len(self.x_train)
        self.n_test = len(self.x_test)
        
        # Evaluate the input zero-coupon bond prices
        self.z_in = price_zero_coupon_bond(self.model.a_param, 
                                        self.model.n_annual_trading_days, 
                                        self.model.r_t_paths[self.t_maturity_idx,
                                                             self.x_train_idxs], 
                                        self.model.sigma, self.expiry, 
                                        self.maturity, self.model.time_0_f_curve, 
                                        self.model.time_0_P_curve, 
                                        self.x_train)
        self.z_in_scale_b = np.mean(self.z_in)
        self.z_in_scale_w = np.std(self.z_in)
        self.z_in_rescaled = (self.z_in - self.z_in_scale_b)/self.z_in_scale_w
        
        # Evaluate exercise at maturity
        V_exercise = np.maximum(price_forward_start_swap(
                                self.model.a_param, self.fixed_rate, 
                                self.model.n_annual_trading_days, 
                                self.notional, 'swap', False, 
                                self.model.r_t_paths[:,self.x_train_idxs], 
                                self.model.sigma, self.swaption_type, 
                                self.tenor_structure, self.expiry, 
                                self.model.time_0_f_curve, 
                                self.model.time_0_P_curve, False, 
                                self.model.x_t_paths[:,self.x_train_idxs]), 
                                0.)
        
        # Assign exercise values as y training data for neural network since 
        # all continuation values are equal to zero for a European swaption
        self.y_train = V_exercise
        self.y_train_scaling_factor = np.mean(self.y_train[self.y_train>0.])
        self.y_train_rescaled = self.y_train/self.y_train_scaling_factor
        
        ## Training and fitting the neural network
        # Initialize and fit the neural network
        if neural_network:
            self.neural_network = neural_network
        else:
            self.neural_network_obj = ShallowFeedForwardNeuralNetwork(
                                            self.swaption_type, 
                                            n_hidden_nodes=self.n_hidden_nodes, 
                                            seed_biases=self.seed_biases, 
                                            seed_weights=self.seed_weights)
            self.neural_network = self.neural_network_obj.neural_network
        
        optimizer_Adamax = keras.optimizers.Adamax(learning_rate=self.learn_rate)
        self.neural_network.compile(loss='MSE', optimizer=optimizer_Adamax)
        
        callback_early_stopping = keras.callbacks.EarlyStopping(monitor='loss', 
                                                        mode='min', verbose=False, 
                                                        patience=100, 
                                                        restore_best_weights=True)
        
        print('Fitting neural network...')
        self.neural_network.fit(self.z_in_rescaled, self.y_train_rescaled, 
                                epochs=self.n_epochs, verbose=False, 
                                callbacks=[callback_early_stopping], 
                                batch_size=self.batch_size)
        # self.neural_network.summary()
        
        # Save the MSE
        y_true = self.y_train_rescaled
        P_in = price_zero_coupon_bond(self.model.a_param, 
                                        self.model.n_annual_trading_days, 
                                        self.model.r_t_paths[self.t_maturity_idx,
                                                             self.x_train_idxs], 
                                        self.model.sigma, 
                                        self.expiry, self.maturity, 
                                        self.model.time_0_f_curve, 
                                        self.model.time_0_P_curve, 
                                        self.x_train)
        x_train_scaled = (P_in - self.z_in_scale_b)/self.z_in_scale_w
        y_pred = self.neural_network.predict(x_train_scaled).flatten()
        self.MSE = np.mean((y_true - y_pred)**2)
        
        if self.test_fit:
            P_in = price_zero_coupon_bond(self.model.a_param, 
                                    self.model.n_annual_trading_days, 
                                    self.model.r_t_paths[self.t_maturity_idx,
                                                         self.x_test_idxs], 
                                    self.model.sigma, self.expiry, 
                                    self.maturity, self.model.time_0_f_curve, 
                                    self.model.time_0_P_curve, self.x_test)
            x_test_scaled = (P_in - self.z_in_scale_b)/self.z_in_scale_w
            y_test = self.neural_network.predict(x_test_scaled).flatten()
            plt.scatter(self.x_test, y_test, color='r', 
                        label='Neural Network Test Data Prediction')
            plt.scatter(self.x_train, y_true, label='Training Data', s=1, 
                        zorder=10)
            plt.scatter(self.x_train, y_pred, label='Training Data Prediction', 
                        color='blue', alpha=.5)
            plt.xlabel('$x_t$')
            plt.title(f'{self.tenor_structure_notation} European ' 
                      + f'{self.swaption_type.capitalize()} Swaption' 
                      + f'\nNeural Network Fit at $T$ = {self.expiry}')
            plt.legend()
            plt.show()
            
        # Save neural network weights
        if self.save_weights is not None:
            print('Saving weights...')
            weight_name = self.weights_dir + '\\European NN Weights' + str(self.index)
            name2 = self.weights_dir + '\\European NN Nodes' + str(self.n_hidden_nodes) + '1F' + str(self.index)
            if self.n_run is not None:
                weight_name = weight_name + 'n_run_' + str(self.n_run)
                name2 = name2 + 'n_run' + str(self.n_run)
            weight_name =  weight_name + '.h5'
            name2 = name2 + '.h5'
            self.neural_network.save_weights(weight_name)
            self.neural_network.save_weights(name2)
            self.index += 1
            
    def construct_ZBO_portfolio(self):
        """
        Info: this method constructs a portfolio of zero-coupon bond options 
            that replicates the European swaption at a given monitor date.
        """
        # Determine zero-coupon bond option payoff type as call or put 
        # corresponding to swaption type (receiver or payer)
        if self.swaption_type.lower() == 'receiver':
            delta_payoff = -1
        elif self.swaption_type.lower() == 'payer':
            delta_payoff = 1
        
        # Obtain the weights and biases of the hidden layer and the weights of 
        # the output layer
        self.weights_hidden = self.neural_network.layers[0].get_weights()[0].flatten()
        self.biases_hidden = self.neural_network.layers[0].get_weights()[1]
        self.weights_output = self.neural_network.layers[1].get_weights()[0].flatten()
        
        # Determine the portfolio strikes
        self.weights_hidden_unscaled = (delta_payoff*self.weights_hidden
                                        /self.z_in_scale_w)
        
        portfolio_strikes = np.zeros(self.n_hidden_nodes)
        for idx in range(self.n_hidden_nodes):
            portfolio_strikes[idx] = (-1*delta_payoff*self.biases_hidden[idx]
                                      /self.weights_hidden_unscaled[idx] 
                                      + self.z_in_scale_b)
            
        # Determine the portfolio weights
        portfolio_weights = (self.weights_output*self.weights_hidden_unscaled
                              *self.y_train_scaling_factor)
        
        # Select the final weights and strikes for the portfolio
        self.positive_strikes_idxs = np.where(portfolio_strikes > 0.)[0]
        final_portfolio_weights = portfolio_weights[self.positive_strikes_idxs]
        final_portfolio_strikes = portfolio_strikes[self.positive_strikes_idxs]
        
        # Store the portfolio weights and strikes
        self.ZBO_portfolios_weights = final_portfolio_weights
        self.ZBO_portfolios_strikes = final_portfolio_strikes
        
    def price_direct_estimator(self,
                               time_t: float,
                               price_error: bool = True
                              ):
        """
        Info: this method evaluates the direct price estimator of the European 
            swaption using the previously constructed portfolio of zero-coupon 
            bond options.
        """
        if time_t > self.tenor_structure[0]:
            raise ValueError('European swaption pricing time may not occur ' 
                             + f'after exercise date {self.expiry}.')
            
        # Price the replicating portfolios using the test data
        if self.swaption_type.lower() == 'payer':
            option_type = 'call'
        elif self.swaption_type.lower() == 'receiver':
            option_type = 'put'
            
        # t_idx = int(time_t*self.model.n_annual_trading_days)
            
        self.replicating_portfolio = np.zeros((len(self.ZBO_portfolios_weights), self.n_test))
        # print(f'portfolio shape: {np.shape(portfolio)}')
        # print(f'strikes: {strikes}')
        
        
        # print(f'w: {w}')
        # print(f'z: {z}')
        # print(f'z_scaled: {z_scaled}')
        
        # z_in = z if w < .000001 else z_scaled
        # # print(f'z_in shape: {np.shape(z_in)}')
        # # z_in = (z - self.z_in_scale_b)/self.z_in_scale_w
        # z_in = z
        # print(f'z_in: {z_in}')
        # print(f'z_in[0]: {z_in[0]}')
        # print(f'shape of z_in: {np.shape(z_in)}')
        
        for count, idx in enumerate(self.positive_strikes_idxs):
            strike = self.ZBO_portfolios_strikes[count]
            weight = self.ZBO_portfolios_weights[count]
            
            ZBO_pathwise_prices = price_zero_coupon_bond_option_exact(
                                            self.model.a_param, self.maturity, 
                                            self.expiry, 
                                            self.model.n_annual_trading_days, 
                                            option_type, None, self.model.sigma, 
                                            strike, time_t, self.model.time_0_f_curve, 
                                            self.model.time_0_P_curve, False, 
                                            self.model.x_t_paths[:,self.x_test_idxs], 
                                            None)
                
            self.replicating_portfolio[count] = (weight*ZBO_pathwise_prices)
            
        self.direct_price_estimator = (np.sum(self.replicating_portfolio, 
                                                        axis=0)
                                       *10**4/self.notional 
                                       if self.units_basis_points 
                                       else np.sum(self.replicating_portfolio, 
                                                              axis=0))
        self.mean_direct_price_estimator = np.mean(self.direct_price_estimator)
        self.se_direct_price_estimator = st.sem(self.direct_price_estimator)
        
        # Evaluate the pathwise absolute errors in the prices with respect to 
        # the exact Bachelier price along with the mean and standard errors
        if price_error:
            self.exact_price = price_European_swaption_exact_Bachelier(
                                                self.model.a_param, self.fixed_rate, 
                                                self.model.n_annual_trading_days, 
                                                self.notional, 
                                                self.model.r_t_paths[:,self.x_test_idxs], 
                                                self.model.sigma, 
                                                self.moneyness, 
                                                self.swaption_type, 
                                                self.tenor_structure, time_t, 
                                                self.model.time_0_f_curve, 
                                                self.model.time_0_rate, 
                                                self.model.time_0_P_curve, 
                                                self.units_basis_points, 
                                                False, 
                                                self.model.x_t_paths[:,self.x_test_idxs])
            
            self.abs_err_direct_price_estimator = (self.direct_price_estimator 
                                                   - self.exact_price)
            self.mean_abs_err_direct_price_estimator = np.mean(self.abs_err_direct_price_estimator)
            self.se_abs_error_direct_price_estimator = st.sem(self.abs_err_direct_price_estimator)
        
    def eval_EPE_profile(self,
                         compare: bool,
                         time_t: float,
                         eval_time_spacing: str = 'monthly',
                         eval_times: list = None,
                         plot_errors: bool = True,
                         r_t_paths: list = None,
                         units_basis_points: bool = False,
                         verbose: bool = False,
                         x_t_paths: list = None,
                        ):
        self.EPE_time_t = time_t
        self.eval_time_spacing = eval_time_spacing

        if r_t_paths is None and x_t_paths is None:
            self.EPE_r_t_paths = self.model.r_t_paths[:,self.x_test_idxs]
            self.EPE_x_t_paths = self.model.x_t_paths[:,self.x_test_idxs]
        elif r_t_paths is not None and x_t_paths is not None:
            self.EPE_r_t_paths = r_t_paths
            self.EPE_x_t_paths = x_t_paths
            
        n_paths = np.shape(self.EPE_x_t_paths)[1]
        
        if eval_times is not None:
            for t in eval_times:
                if t>self.expiry:
                    raise ValueError('EPE evaluation times may not exceed expiry.')
                    
            # Add check for dimensions of eval_times...
            self.EPE_eval_times = eval_times[eval_times<self.expiry]
            self.n_eval_times = len(eval_times)
            
        else:
            if self.eval_time_spacing.lower() == 'daily':
                self.n_eval_times = int(self.model.n_annual_trading_days*self.expiry)
            elif self.eval_time_spacing.lower() == 'weekly':
                self.n_eval_times = int(52*self.expiry)
            elif self.eval_time_spacing.lower() == 'biweekly':
                self.n_eval_times = int(26*self.expiry)
            elif self.eval_time_spacing.lower() == 'monthly':
                self.n_eval_times = int(12*self.expiry)
                
            ## Evaluate swaption exposures. Due to the positive part operator in 
            ## the swaption payoff function, the exposures are simply equal to the 
            ## swaption prices at the evaluation dates discounted back to time_t.
            self.EPE_eval_times = np.linspace(self.EPE_time_t, 
                                              self.tenor_structure[0], 
                                              self.n_eval_times, endpoint=False)
            
        self.EPE_RLNN = np.zeros((self.n_eval_times, n_paths))
        
        for count, eval_t in enumerate(self.EPE_eval_times):
            if verbose:
                print(f'Evaluating swaption EPE at time {eval_t}...')
            self.price_direct_estimator(eval_t, price_error=False)
            self.EPE_RLNN[count] = np.maximum(self.direct_price_estimator, 0)
            
        if compare:
            model_copy = copy.deepcopy(self.model)
            model_copy.r_t_paths = self.EPE_r_t_paths
            model_copy.x_t_paths = self.EPE_x_t_paths
            Bachelier_instance = EuropeanSwaption(self.fixed_rate, model_copy, 
                                                  self.moneyness,
                                                  self.notional, 
                                                  self.swaption_type, 
                                                  self.tenor_structure, 
                                                  self.time_t, 
                                                  self.units_basis_points)
            Bachelier_instance.eval_exposures_Bachelier(plot=False, 
                                            time_t=self.EPE_time_t,
                                            eval_times=self.EPE_eval_times)
            self.EPE_Bachelier = np.maximum(Bachelier_instance.EPE_array, 
                                                             0)
            
    def write_EPE_profile(self,
                          compare: bool,
                          mean: bool = True, 
                          units_basis_points: bool = False
                         ):
        if not self.units_basis_points:
            adjust_units = 10000/self.notional if units_basis_points else 1
            
        if mean is True:
            file_dir_and_name = str(self.model.data_dir + 'Mean_of_' 
                                    + self.tenor_structure_notation 
                                    + '_Bermudan_' + self.swaption_type 
                                    + '_swaption_' + 'RLNN_EPE_profiles')
            np.savetxt(file_dir_and_name + '.txt', 
                       np.array((self.EPE_eval_times, 
                                 adjust_units*np.mean(self.EPE_RLNN, axis=-1))), 
                       delimiter=', ', fmt='%.10g', 
                       header=str(f'{np.shape(self.EPE_RLNN)[-1]:,} Monte Carlo paths. ' 
                                  + 'First line: time (years); second line: mean EPE profile'))
            print('Mean RLNN EPE profike was saved to '
                  + f'{self.model.data_dir}\Mean_of_{file_dir_and_name}.')
            
            file_dir_and_name = str(self.model.data_dir + 'SEM_of_' 
                                    + self.tenor_structure_notation 
                                    + '_Bermudan_' + self.swaption_type 
                                    + '_swaption_' + 'RLNN_EPE_profiles')
            np.savetxt(file_dir_and_name + '.txt', 
                       np.array((self.EPE_eval_times, 
                                 st.sem(adjust_units*self.EPE_RLNN, axis=-1))), 
                       delimiter=', ', fmt='%.10g', 
                       header=str(f'{np.shape(self.EPE_RLNN)[-1]:,} Monte Carlo paths. ' 
                                  + 'First line: time (years); second line: SEM of EPE profile'))
            print('SEM of RLNN EPE profiles were saved to '
                  + f'{self.model.data_dir}\SEM_of_{file_dir_and_name}.')
            
            if compare:
                file_dir_and_name = str(self.model.data_dir + 'Mean_of_' 
                                        + self.tenor_structure_notation 
                                        + '_Bermudan_' + self.swaption_type 
                                        + '_swaption_' + 'Bachelier_EPE_profiles')
                np.savetxt(file_dir_and_name + '.txt', 
                           np.array((self.EPE_eval_times, 
                                     adjust_units*np.mean(self.EPE_Bachelier, axis=-1))), 
                           delimiter=', ', fmt='%.10g', 
                           header=str(f'{np.shape(self.EPE_Bachelier)[-1]:,} Monte Carlo paths. ' 
                                      + 'First line: time (years); second line: mean EPE profile'))
                print('Mean sensitivities were saved to '
                      + f'{self.model.data_dir}\Mean_of_{file_dir_and_name}.')
                
                file_dir_and_name = str(self.model.data_dir + 'SEM_of_' 
                                        + self.tenor_structure_notation 
                                        + '_Bermudan_' + self.swaption_type 
                                        + '_swaption_' + 'Bachelier_EPE_profiles')
                np.savetxt(file_dir_and_name + '.txt', 
                           np.array((self.EPE_eval_times, 
                                     st.sem(adjust_units*self.EPE_Bachelier, axis=-1))), 
                           delimiter=', ', fmt='%.10g', 
                           header=str(f'{np.shape(self.EPE_Bachelier)[-1]:,} Monte Carlo paths. ' 
                                      + 'First line: time (years); second line: SEM of EPE profile'))
                print('SEM of sensitivities were saved to '
                      + f'{self.model.data_dir}\SEM_of_{file_dir_and_name}.')
                
        elif mean.lower() == 'both':
            # Save RLNN profile
            file_dir_and_name = str(self.model.data_dir
                                    + self.tenor_structure_notation 
                                    + '_Bermudan_' + self.swaption_type 
                                    + '_swaption_' + 'RLNN_EPE_profile.parquet')
            write_Parquet_data(self.model.experiment_dir, file_dir_and_name, 
                           self.EPE_RLNN)
            print('RLNN EPE profiles were saved to '
                  + f'{self.model.data_dir}\{file_dir_and_name}.')
            
            # Save Bachelier profile
            file_dir_and_name = str(self.model.data_dir
                                    + self.tenor_structure_notation 
                                    + '_Bermudan_' + self.swaption_type 
                                    + '_swaption_' + 'Bachelier_EPE_profile.parquet')
            write_Parquet_data(self.model.experiment_dir, file_dir_and_name, 
                           self.EPE_Bachelier)
            print('Bachelier EPE profiles were saved to '
                  + f'{self.model.data_dir}\{file_dir_and_name}.')
            
            # Save mean and SEM of profiles
            file_dir_and_name = str(self.model.data_dir + 'Mean_of_' 
                                    + self.tenor_structure_notation 
                                    + '_Bermudan_' + self.swaption_type 
                                    + '_swaption_' + 'RLNN_EPE_profiles')
            np.savetxt(file_dir_and_name + '.txt', 
                       np.array((self.EPE_eval_times, 
                                 adjust_units*np.mean(self.EPE_RLNN, axis=-1))), 
                       delimiter=', ', fmt='%.10g', 
                       header=str(f'{np.shape(self.EPE_RLNN)[-1]:,} Monte Carlo paths. ' 
                                  + 'First line: time (years); second line: mean EPE profile'))
            print('Mean RLNN EPE profike was saved to '
                  + f'{self.model.data_dir}\Mean_of_{file_dir_and_name}.')
            
            file_dir_and_name = str(self.model.data_dir + 'SEM_of_' 
                                    + self.tenor_structure_notation 
                                    + '_Bermudan_' + self.swaption_type 
                                    + '_swaption_' + 'RLNN_EPE_profiles')
            np.savetxt(file_dir_and_name + '.txt', 
                       np.array((self.EPE_eval_times, 
                                 st.sem(adjust_units*self.EPE_RLNN, axis=-1))), 
                       delimiter=', ', fmt='%.10g', 
                       header=str(f'{np.shape(self.EPE_RLNN)[-1]:,} Monte Carlo paths. ' 
                                  + 'First line: time (years); second line: SEM of EPE profile'))
            print('SEM of RLNN EPE profiles were saved to '
                  + f'{self.model.data_dir}\SEM_of_{file_dir_and_name}.')
            
            if compare:
                file_dir_and_name = str(self.model.data_dir + 'Mean_of_' 
                                        + self.tenor_structure_notation 
                                        + '_Bermudan_' + self.swaption_type 
                                        + '_swaption_' + 'Bachelier_EPE_profiles')
                np.savetxt(file_dir_and_name + '.txt', 
                           np.array((self.EPE_eval_times, 
                                     adjust_units*np.mean(self.EPE_Bachelier, axis=-1))), 
                           delimiter=', ', fmt='%.10g', 
                           header=str(f'{np.shape(self.EPE_Bachelier)[-1]:,} Monte Carlo paths. ' 
                                      + 'First line: time (years); second line: mean EPE profile'))
                print('Mean sensitivities were saved to '
                      + f'{self.model.data_dir}\Mean_of_{file_dir_and_name}.')
                
                file_dir_and_name = str(self.model.data_dir + 'SEM_of_' 
                                        + self.tenor_structure_notation 
                                        + '_Bermudan_' + self.swaption_type 
                                        + '_swaption_' + 'Bachelier_EPE_profiles')
                np.savetxt(file_dir_and_name + '.txt', 
                           np.array((self.EPE_eval_times, 
                                     st.sem(adjust_units*self.EPE_Bachelier, axis=-1))), 
                           delimiter=', ', fmt='%.10g', 
                           header=str(f'{np.shape(self.EPE_Bachelier)[-1]:,} Monte Carlo paths. ' 
                                      + 'First line: time (years); second line: SEM of EPE profile'))
                print('SEM of sensitivities were saved to '
                      + f'{self.model.data_dir}\SEM_of_{file_dir_and_name}.')
        elif mean is False:
            # Save RLNN profiles
            file_dir_and_name = str(self.model.data_dir
                                    + self.tenor_structure_notation 
                                    + '_Bermudan_' + self.swaption_type 
                                    + '_swaption_' + 'RLNN_EPE_profile.parquet')
            write_Parquet_data(self.model.experiment_dir, file_dir_and_name, 
                           self.EPE_RLNN)
            print('RLNN EPE profiles were saved to '
                  + f'{self.model.data_dir}\{file_dir_and_name}.')
            
            if compare:
                # Save Bachelier profiles
                file_dir_and_name = str(self.model.data_dir
                                        + self.tenor_structure_notation 
                                        + '_Bermudan_' + self.swaption_type 
                                        + '_swaption_' + 'Bachelier_EPE_profile.parquet')
                write_Parquet_data(self.model.experiment_dir, file_dir_and_name, 
                               self.EPE_Bachelier)
                print('Bachelier EPE profiles were saved to '
                      + f'{self.model.data_dir}\{file_dir_and_name}.')
            
    def plot_EPE_profile(self, 
                         plot: str,
                         save_plot: bool = True,
                         tenors: list = None,
                         time_frame: list = None,
                         units_basis_points: bool = False
                        ):
        # Logical checks
        
           
        ## Plot results
        fig, ax = plt.subplots()
        plt.suptitle('Expected Positive Exposure Profile of\n' 
                     + f'{self.tenor_structure_notation} ' 
                     + f'European {self.swaption_type.capitalize()} ' 
                     + 'Swaption ' + f'{int(self.moneyness*100)}% Moneyness')
        
        # Adjust plotting if data is in absolute units but plotting 
        # must be in basis points of the notional
        if units_basis_points and not self.units_basis_points:
            adjust_units = 10000/self.notional
        else:
            adjust_units = 1
        if plot.lower() == 'rlnn':
            ax.plot(self.EPE_eval_times, np.mean(self.EPE_RLNN, axis=1)*adjust_units, 
                   label='Replicating\nPortfolio', 
                   linewidth=1.5)
            ax.fill_between(self.EPE_eval_times, 
                            (np.mean(self.EPE_RLNN, axis=1) 
                             + 1.96*st.sem(self.EPE_RLNN, axis=1))
                            *adjust_units, 
                            (np.mean(self.EPE_RLNN, axis=1) 
                             - 1.96*st.sem(self.EPE_RLNN, axis=1))
                            *adjust_units, 
                            alpha=.5)
        elif plot.lower() == 'compare':
            ax.plot(self.EPE_eval_times, np.mean(self.EPE_RLNN, axis=1)*adjust_units, 
                   label='Replicating\nPortfolio')
            ax.plot(self.EPE_eval_times, np.mean(self.EPE_Bachelier, axis=1)*adjust_units, 
                   label='Bachelier')
            ax.fill_between(self.EPE_eval_times, 
                            (np.mean(self.EPE_RLNN, axis=1) 
                             + 1.96*st.sem(self.EPE_RLNN, axis=1))
                            *adjust_units, 
                            (np.mean(self.EPE_RLNN, axis=1)
                             - 1.96*st.sem(self.EPE_RLNN, axis=1))
                            *adjust_units, 
                            alpha=.5)
            # ax.fill_between(self.EPE_eval_times, 
            #                 (np.mean(self.EPE_RLNN, axis=1) 
            #                  + 1.96*st.sem(self.EPE_RLNN, axis=1))
            #                 *adjust_units, 
            #                 (np.mean(self.EPE_RLNN, axis=1) 
            #                  - 1.96*st.sem(self.EPE_RLNN, axis=1))
            #                 *adjust_units, 
            #                 alpha=.5)
            
        # Set axis labels
        ax.set_xlabel('Time $t$ (years)')
        if units_basis_points or self.units_basis_points:
            ax.set_ylabel('EPE$(t)$')
        else:
            ax.set_ylabel('EPE$(t)$ (basis points of the notional)')
        
        plt.ylim(top=plt.yticks()[0][-1])
        ax.legend(loc='best')
        
        plt.tight_layout()
        
        if save_plot:
            plot_name = (f'{self.tenor_structure_notation} European ' 
                         + f'{self.swaption_type.capitalize()} Swaption EPE ' 
                         + f'{int(self.moneyness)*100} moneynesss plot')
            file_dir_and_name = str(self.model.figures_dir + 'Mean_of_' 
                                    + (plot_name.replace(' ','_'))  + '-' 
                        + datetime.now().strftime('%Y-%m-%d_%H%M%S'))
            plt.savefig(file_dir_and_name, bbox_inches='tight')
            print('\nPlot was saved to ' + file_dir_and_name + '.png')
            
        
        
        plt.show()
            
    def eval_forward_sensitivities(self, 
                                   eval_times: list,
                                   tenors: str,
                                   Bachelier: bool = True,
                                   bump_size: float = .0001,
                                   plot: bool = False,
                                   r_t_paths: list = None,
                                   x_t_paths: list = None
                           ):
        """
        Info: currenly implemented for time-zero Deltas only...
        """
        ISDA_SIMM_tenors = np.array(list(self.model.ISDA_SIMM_tenors_dict.values()))
        self.bump_size = bump_size
        
        ## Input checks
        # Check whether short rate and zero-mean process paths were passed
        if r_t_paths is None and x_t_paths is not None:
            raise ValueError('pass corresponding short rate and zero-mean ' 
                             + 'process paths for determination of ' 
                             + 'pathwise stopping times')
        elif r_t_paths is not None and x_t_paths is None:
            raise ValueError('pass corresponding short rate and zero-mean ' 
                             + 'process paths for determination of ' 
                             + 'pathwise stopping times')
        elif r_t_paths is None and x_t_paths is None:
            self.sensitivities_r_t_paths = self.model.r_t_paths[:,self.x_test_idxs] # self.model.r_t_paths[:,self.test_idxs_arr[0]]
            self.sensitivities_x_t_paths = self.model.x_t_paths[:,self.x_test_idxs] # self.model.x_t_paths[:,self.test_idxs_arr[0]]
            n_paths = self.n_test
        else:
            self.sensitivities_r_t_paths = r_t_paths
            self.sensitivities_x_t_paths = x_t_paths
            n_paths = np.shape(x_t_paths)[1]
            
        # Check whether tenor time(s) passed as string or float(s)
        if type(tenors) is str:
            if tenors.lower() == 'isda-simm':
                # Select the relevant ISDA-SIMM tenors, i.e., those that occur 
                # during the swaption lifetime
                max_idx = np.searchsorted(ISDA_SIMM_tenors, self.maturity)
                self.Delta_tenors = ISDA_SIMM_tenors[:max_idx+1]
            else:
                raise ValueError('forward Delta sensitivity evaluation times ' 
                                 + 'not recognized. Enter as "ISDA-SIMM" to ' 
                                 + 'evaluate on all ISDA-SIMM tenors occuring ' 
                                 + 'during the swaption lifetime or enter as ' 
                                 + 'a float or list of floats with values ' 
                                 + 'equal to or smaller than last monitor date.')
        elif tenors is None:
            tenors = np.array([])
        elif not isinstance(tenors, np.ndarray):
            tenors = np.array(tenors)
            
            # Check whether last tenor time occurs after swaption expiry
            if tenors[-1] > self.maturity:
                raise ValueError('forward Delta sensitivity evaluation times ' 
                                 + f'may not exceed swaption maturity T={self.maturity}')
                
            # Add relevant ISDA-SIMM tenors to Delta tenors
            ISDA_SIMM_tenors = np.array(list(self.model.ISDA_SIMM_tenors_dict.values()))
            self.Delta_tenors = tenors
        n_Delta_tenors = len(self.Delta_tenors)
        
        # Evaluation times
        if not isinstance(eval_times, np.ndarray):
            eval_times = np.array([eval_times])
        if (eval_times>self.expiry).any():
            raise ValueError('forward Delta evaluation times may not occur ' 
                             + f'after swaption expiry {self.expiry}')
        self.Delta_eval_times = eval_times[eval_times<self.expiry]
        n_eval_times = len(self.Delta_eval_times)
        
        # Initialize the dVdR array
        self.dVdR_RLNN = np.zeros((n_eval_times, n_Delta_tenors, n_paths))
        
        ## Compute forward deltas
        # Loop over evaluation times
        for eval_count, eval_time in enumerate(self.Delta_eval_times):
            # print(f'\n\nEvaluating forward dVdR at time {eval_time}...')
            
            t_idx = int(eval_time*self.model.n_annual_trading_days)
            P_t_m = price_zero_coupon_bond(self.model.a_param, 
                                           self.model.n_annual_trading_days, 
                                           self.sensitivities_r_t_paths[t_idx], 
                                           self.model.sigma, eval_time, 
                                           self.expiry, 
                                           self.model.time_0_f_curve, 
                                           self.model.time_0_P_curve, 
                                           self.sensitivities_x_t_paths[t_idx])
            P_t_M = price_zero_coupon_bond(self.model.a_param, 
                                           self.model.n_annual_trading_days, 
                                           self.sensitivities_r_t_paths[t_idx], 
                                           self.model.sigma, 
                                           eval_time, self.maturity, 
                                           self.model.time_0_f_curve, 
                                           self.model.time_0_P_curve, 
                                           self.sensitivities_x_t_paths[t_idx])

            # Loop over tenors
            for tenor_count, tenor in enumerate(self.Delta_tenors):
                # print(f'\nEvaluating time {eval_time} dVdR_k for k={tenor}')
                
                
                # Initialize fractions and zero-coupon bond prices
                alpha_m = 0.
                alpha_M = 0.
                
                # Evaluate P_t_m
                tau_m = self.expiry - eval_time
                
                if tau_m in ISDA_SIMM_tenors:
                    if tenor == tau_m:
                        alpha_m = 1.
                else:
                    left_bucket = ISDA_SIMM_tenors[ISDA_SIMM_tenors<tau_m][-1]
                    right_bucket = ISDA_SIMM_tenors[ISDA_SIMM_tenors>tau_m][0]
                    
                    if tenor == left_bucket or tenor == right_bucket:
                        alpha = (right_bucket - tau_m)/(right_bucket - left_bucket)
                        alpha_m = alpha if tenor == left_bucket else 1 - alpha
                        
                # Evaluate P_t_M
                tau_M = self.maturity - eval_time
                # print(f'tau_M: {tau_M}')
                
                if tau_M in ISDA_SIMM_tenors:
                    if tenor == tau_M:
                        alpha_M = 1.
                        
                else:
                    left_bucket = ISDA_SIMM_tenors[ISDA_SIMM_tenors<tau_M][-1]
                    right_bucket = ISDA_SIMM_tenors[ISDA_SIMM_tenors>tau_M][0]
                    
                    if tenor == left_bucket or tenor == right_bucket:
                        alpha = (right_bucket - tau_M)/(right_bucket - left_bucket)
                        alpha_M = alpha if tenor == left_bucket else 1 - alpha
                    
                # print(f'tau_m: {tau_m}')
                # print(f'P_t_m: {P_t_m}')
                    
                ## Evaluate the dVdR_k vector with k the current ISDA-SIMM 
                ## tenor
                # Select the weights and strikes of the replicating portfolio
                weights = self.ZBO_portfolios_weights
                strikes = self.ZBO_portfolios_strikes
                
                # Initialize the dVdR_k vector. 
                # NOTE: move to its own function in bonds_and_bond_options???
                dVdR_k = np.zeros((len(strikes), n_paths))
                # print(f'np.shape(dVdR_k): {np.shape(dVdR_k)}')
                
                for strike_count, strike in enumerate(strikes):
                    weight = weights[strike_count]
                    B_Tm_TM = B_func(self.model.a_param, self.expiry, self.maturity)
                    sigma_p = self.model.sigma*np.sqrt((1 - np.exp(-2*self.model.a_param*(self.expiry - eval_time)))
                                            /(2*self.model.a_param))*B_Tm_TM
                        
                    if 0. in P_t_m or 0. in P_t_M:
                        h = 0.
                    else:
                        h = 1/sigma_p*np.log(P_t_M/(P_t_m*strike)) + sigma_p/2
                    delta_payoff = 1 if self.swaption_type == 'payer' else -1
                    
                    # print(f'Number of zero-value P_t_m: {np.where(P_t_m==0.)}')
                    
                    dVdR_k[strike_count] = (bump_size*weight
                                            *(-1*delta_payoff*st.norm.cdf(delta_payoff*h)
                                              *alpha_M*tau_M*P_t_M 
                                              + delta_payoff*strike*
                                              st.norm.cdf(delta_payoff*(h - sigma_p))
                                              *alpha_m*tau_m*P_t_m))
                   # if np.isnan(np.mean(np.sum(dVdR_k, axis=0))):
                        # print(f'NaN!!! np.mean(np.sum(dVdR_k, axis=0))={np.mean(np.sum(dVdR_k, axis=0))}')
                        
                
                self.dVdR_RLNN[eval_count,tenor_count] = np.sum(dVdR_k, axis=0)
                
        if Bachelier:
            BachelierModel = EuropeanSwaption(self.fixed_rate, self.model, 
                                              self.moneyness, self.notional, 
                                              self.swaption_type, 
                                              self.tenor_structure, 
                                              self.time_t, 
                                              self.units_basis_points)
            BachelierModel.eval_forward_sensitivities_Bachelier(bump_size, 
                                                    eval_times, tenors, 
                                                    self.sensitivities_r_t_paths, 
                                                    self.sensitivities_x_t_paths)
            
            self.dVdR_Bachelier = BachelierModel.dVdR_Bachelier
                
        # Adjust units to basis points of the notional
        if self.units_basis_points:
            self.dVdR_RLNN = self.dVdR_RLNN*0.0001/self.notional
            
    # def eval_forward_sensitivities_Bachelier(self,
    #                                          eval_times: list,
    #                                          tenors: str,
    #                                          bump_size: float = .0001,
    #                                          plot: bool = False,
    #                                          r_t_paths: list = None,
    #                                          x_t_paths: list = None
    #                                         ):
        
        
    
    def plot_forward_sensitivities(self, 
                                   plot: str,
                                   plot_errors: bool = True,
                                   save_plot: bool = True,
                                   tenors: list = None,
                                   units_basis_points: bool = False
                                  ):
        # Check whether forward sensitivities were previously evaluated
        if plot.lower() == 'rlnn' and not hasattr(self, 'dVdR_RLNN'):
            raise ValueError('RLNN forward sensitivities have not yet been evaluated.')
        if plot.lower() == 'bachelier' and not hasattr(self, 'dVdR_Bachelier'):
            raise ValueError('Bachelier forward sensitivities have not yet been evaluated.')
        elif (plot.lower() == 'compare' and not (hasattr(self, 'dVdR_RLNN')
                                                 or hasattr(self, 'dVdR_Bachelier'))):
            raise ValueError('Bachelier forward sensitivities have not yet been evaluated.')
            
        # If passed, only plot the tenors specified by the user
        if isinstance(tenors, list):
            if all(tenor in self.Delta_tenors for tenor in tenors):
                tenor_idxs = np.searchsorted(self.Delta_tenors, tenors)#np.where(self.Delta_tenors==tenors)[0]
                print(f'tenor_idxs: {tenor_idxs}')
                print(f'self.Delta_tenors[tenor_idxs]: {self.Delta_tenors[tenor_idxs]}')
            else:
                raise ValueError('plotting tenors must be contained in the ' 
                                 + 'set of evaluated tenors '
                                 + f'{self.Delta_tenors}.')
        else:
            tenor_idxs = np.arange(0, len(self.Delta_tenors))
            
        ## Plot results
        fig, ax = plt.subplots()
        if plot.lower() == 'rlnn':
            plt.suptitle(r'$\frac{\partial \hat{{V}}}{\partial R_k}$ of ' 
                         + 'Replicating Portfolio of\n'
                         + f'{self.tenor_structure_notation} ' 
                         + f'European {self.swaption_type.capitalize()} ' 
                         + 'Swaption')
        elif plot.lower() == 'compare':
            plt.suptitle(r'$\frac{\partial V}{\partial R_k}$ of ' 
                         + f'{self.tenor_structure_notation} ' 
                         + f'European {self.swaption_type.capitalize()} ' 
                         + 'Swaption')
        
        line_styles = ['dotted', 'solid', 'dashed']
        n_repeats = 4

        for count, tenor in enumerate(self.Delta_tenors[tenor_idxs]):
            tenor_count = tenor_idxs[count]
            
            if tenor in list(self.model.ISDA_SIMM_tenors_dict.values()):
                # list(mydict.keys())[list(mydict.values()).index(16)]
                label_tenor = f'{list(self.model.ISDA_SIMM_tenors_dict.keys())[list(self.model.ISDA_SIMM_tenors_dict.values()).index(tenor)]}'
                label = label_tenor if tenor < 1. else label_tenor.upper()
            else:
                label = str(tenor)
            label = '$k$='+label
            
            # Adjust plotting if data is in absolute units but plotting 
            # must be in basis points of the notional
            if units_basis_points and not self.units_basis_points:
                adjust_units = 10000/self.notional
            else:
                adjust_units = 1
                
            if plot.lower() == 'rlnn':
                ax.plot(self.Delta_eval_times, np.mean(self.dVdR_RLNN[:,tenor_count,:], axis=1)*adjust_units, 
                       label=label, linestyle=line_styles[tenor_count//n_repeats], 
                       linewidth=1.5)
                if plot_errors:
                    ax.fill_between(self.Delta_eval_times, 
                                    (np.mean(self.dVdR_RLNN[:,tenor_count,:], axis=1) 
                                     + 1.96*st.sem(self.dVdR_RLNN[:,tenor_count,:], axis=1))
                                    *adjust_units, 
                                    (np.mean(self.dVdR_RLNN[:,tenor_count,:], axis=1) 
                                     - 1.96*st.sem(self.dVdR_RLNN[:,tenor_count,:], axis=1))
                                    *adjust_units, 
                                    alpha=.5)
            elif plot.lower() == 'compare':
                ax.plot(self.Delta_eval_times, np.mean(self.dVdR_RLNN[:,tenor_count,:], axis=1)*adjust_units, 
                       label='Replicating\nPortfolio' if count == 0 else None, 
                       linestyle='solid', color='C0', linewidth=1.5)
                ax.plot(self.Delta_eval_times, np.mean(self.dVdR_Bachelier[:,tenor_count,:], axis=1)*adjust_units, 
                       label='Bachelier' if count == 0 else None, 
                       linestyle='dashed', color='C1', linewidth=1.5)
                if plot_errors:
                    ax.fill_between(self.Delta_eval_times, 
                                    (np.mean(self.dVdR_RLNN[:,tenor_count,:], axis=1) 
                                     + 1.96*st.sem(self.dVdR_RLNN[:,tenor_count,:], axis=1))
                                    *adjust_units, 
                                    (np.mean(self.dVdR_RLNN[:,tenor_count,:], axis=1) 
                                     - 1.96*st.sem(self.dVdR_RLNN[:,tenor_count,:], axis=1))
                                    *adjust_units, 
                                    alpha=.5)
                    ax.fill_between(self.Delta_eval_times, 
                                    (np.mean(self.dVdR_Bachelier[:,tenor_count,:], axis=1) 
                                     + 1.96*st.sem(self.dVdR_Bachelier[:,tenor_count,:], axis=1))
                                    *adjust_units, 
                                    (np.mean(self.dVdR_Bachelier[:,tenor_count,:], axis=1) 
                                     - 1.96*st.sem(self.dVdR_Bachelier[:,tenor_count,:], axis=1))
                                    *adjust_units, 
                                    alpha=.5)
                    
        ax.set_xlabel('Time $t$ (years)')
        if units_basis_points or self.units_basis_points:
            ax.set_ylabel('Basis Points of the Notional')
        
        plt.ylim(top=plt.yticks()[0][-1])
        
        # Add legend (credit to Joe Kington at 
        # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot)
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        
        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.ylim(top=plt.yticks()[0][-1])
        plt.ylim(bottom=plt.yticks()[0][0])
        
        plt.tight_layout()
        
        if save_plot:
            plot_name = f'{self.tenor_structure_notation} European {self.swaption_type.capitalize()} Swaption dVdR plot'
            file_dir_and_name = str(self.model.figures_dir + 'Mean_of_' 
                                    + (plot_name.replace(' ','_'))  + '-' 
                        + datetime.now().strftime('%Y-%m-%d_%H%M%S'))
            plt.savefig(file_dir_and_name, bbox_inches='tight')
            print('\nPlot was saved to ' + file_dir_and_name + '.png')
        
        plt.show()
        
    def write_sensitivities(self,
                            # dVdR: str,
                            dVdR_type: str = 'RLNN',
                            mean: bool = True, 
                            units_basis_points: bool = False
                           ):
        
        if dVdR_type.lower() == 'rlnn':
            if not hasattr(self, 'dVdR_RLNN'):
                raise ValueError('RLNN sensitivities not yet evaluated.')
            else:
                dVdR = self.dVdR_RLNN
        elif dVdR_type.lower() == 'bachelier':
            if not hasattr(self, 'dVdR_Bachelier'):
                raise ValueError('Bachelier sensitivities not yet evaluated.')
            else:
                dVdR = self.dVdR_Bachelier
        else:
            raise ValueError('dVdR_type not recognized. Enter as "RLNN" or "Bachelier".')
        
        if not self.units_basis_points:
            adjust_units = 10000/self.notional if units_basis_points else 1
        
        for tenor_count, tenor in enumerate(self.Delta_tenors):
            label_tenor = f'{list(self.model.ISDA_SIMM_tenors_dict.keys())[list(self.model.ISDA_SIMM_tenors_dict.values()).index(tenor)]}'
            file_dir_and_name = str(self.model.data_dir
                                    + self.tenor_structure_notation 
                                    + 'European' + self.swaption_type 
                                    + '_swaption_' + f'{dVdR_type}_' 
                                    + 'sensitivities_k=' 
                                    + label_tenor + '.parquet')
            if mean is True:
                file_dir_and_name = str(self.model.data_dir + 'Mean_of_' 
                                        + self.tenor_structure_notation 
                                        + '_European_' + self.swaption_type 
                                        + '_swaption_' + f'{dVdR_type}_' 
                                        + 'sensitivities_k=' 
                                        + label_tenor)# + '.parquet')
                np.savetxt(file_dir_and_name + '.txt', 
                           np.array((self.Delta_eval_times, 
                                     adjust_units*np.mean(dVdR[:,tenor_count,:], 
                                                          axis=-1))), 
                           delimiter=', ', fmt='%.10g', 
                           header=str(f'{np.shape(dVdR)[-1]:,} Monte Carlo paths. ' 
                                      + 'First line: time (years); second line: mean sensitivity'))
                print('Mean sensitivities were saved to '
                      + f'{self.model.data_dir}\Mean_of_{file_dir_and_name}.')
                
                file_dir_and_name = str(self.model.data_dir + 'SEM_of_' 
                                        + self.tenor_structure_notation 
                                        + '_European_' + self.swaption_type 
                                        + '_swaption_' + f'{dVdR_type}_' 
                                        + 'sensitivities_k=' 
                                        + label_tenor)# + '.parquet')
                np.savetxt(file_dir_and_name + '.txt', 
                           np.array((self.Delta_eval_times, 
                                     st.sem(adjust_units*dVdR[:,tenor_count,:], 
                                            axis=-1))), 
                           delimiter=', ', fmt='%.10g', 
                           header=str(f'{np.shape(dVdR)[-1]:,} Monte Carlo paths. ' 
                                      + 'First line: time (years); second line: SEM of sensitivity'))
                print('SEM of sensitivities were saved to '
                      + f'{self.model.data_dir}\SEM_of_{file_dir_and_name}.')
            elif mean.lower() == 'both':
                write_Parquet_data(self.model.experiment_dir, file_dir_and_name, 
                               dVdR[:,tenor_count,:])
                print('Sensitivities were saved to '
                      + f'{self.model.data_dir}\\{file_dir_and_name}.')
                
                file_dir_and_name = str(self.model.data_dir + 'Mean_of_' 
                                        + self.tenor_structure_notation 
                                        + '_European_' + self.swaption_type 
                                        + '_swaption_' + f'{dVdR_type}_' 
                                        + 'sensitivities_k=' 
                                        + label_tenor)# + '.parquet')
                np.savetxt(file_dir_and_name + '.txt', 
                           np.array((self.Delta_eval_times, 
                                     np.mean(adjust_units*dVdR[:,tenor_count,:], 
                                             axis=-1))), 
                           delimiter=', ', fmt='%.10g', 
                           header=str(f'{np.shape(dVdR)[-1]:,} Monte Carlo paths. ' 
                                      + 'First line: time (years); second line: mean sensitivity'))
                print('Mean sensitivities were saved to '
                      + f'{self.model.data_dir}\Mean_of_{file_dir_and_name}.')
                
                file_dir_and_name = str(self.model.data_dir + 'SEM_of_' 
                                        + self.tenor_structure_notation 
                                        + '_European_' + self.swaption_type 
                                        + '_swaption_' + f'{dVdR_type}_' 
                                        + 'sensitivities_k=' 
                                        + label_tenor)# + '.parquet')
                np.savetxt(file_dir_and_name + '.txt', 
                           np.array((self.Delta_eval_times, 
                                     st.sem(adjust_units*dVdR[:,tenor_count,:], 
                                            axis=-1))), 
                           delimiter=', ', fmt='%.10g', 
                           header=str(f'{np.shape(dVdR)[-1]:,} Monte Carlo paths. ' 
                                      + 'First line: time (years); second line: SEM of sensitivity'))
                print('SEM of sensitivities were saved to '
                      + f'{self.model.data_dir}\SEM_of_{file_dir_and_name}.')
            elif mean is False:
                write_Parquet_data(self.model.experiment_dir, file_dir_and_name, 
                               dVdR[:,tenor_count,:])
                print('Sensitivities were saved to '
                      + f'{self.model.data_dir}\{file_dir_and_name}.')
                
    def visualize_neural_network(self, 
                                 file_name: str
                                ):
        visualize_neural_network(self.model.experiment_dir, file_name, 
                                 self.neural_network)
        
class Swap:
    """
    This class represents a forward start interest swap. 
    
    Attributes: 
        model: an instance of the OneFactorHullWhiteModel class containing the 
            fundamental interest rate curves and simulated short rates.
            
    Methods:
        price_forward_start_swap(): computes the price of the swap.
    """
    
    def __init__(self,
                 model: object
                ):
        self.model = model
        
    def price_forward_start_swap(self,
                                 fixed_rate: float,
                                 moneyness: float,
                                 notional: float,
                                 swap_type: str,
                                 tenor_structure: list,
                                 time_t: float, 
                                 units_basis_points: bool,
                                 payoff_var: str = 'swap',
                                 plot_timeline: bool = False,
                                 verbose: bool = False
                                ):
        """
        Info: this method prices a forward start interest rate swap (IRS) at 
            time t using the market time-zero zero-coupon bond curve for t = 0 
            and by constructing the zero-coupon bond curves for t > 0 using the 
            simulated short rates.
        
        Input:
            fixed_rate: the fixed interest rate of the forward start IRS.
            
            notional: the notional amount of the forward start IRS.
            
            payoff_var: if 'swap', the swap payoff is determined using the forward 
                swap rate; if 'forward', the swap payoff is determined using the 
                simply compounded forward rate.
            
            plot_timeline: if True, the forward start IRS timeline is plotted 
                and saved to the local folder.
                
            moneyness: if None, the value of the forward start IRS 
                is computed using the simply-compounded forward rates; if 
                passed as a float, the value of the IRS is computed by setting 
                the fixed rate equal to the forward swap rate multiplied by the 
                passed float.
                
            swap_type: a string 'payer' or 'receiver' specifying whether the 
                forward start IRS is a payer or receiver IRS.
                
            tenor_structure: a list containing the forward IRS starting date as 
                the first entry and the payment dates as the remaining entries.
                
            time_t: the time of valuation.
            
            units_basis_points: if True, the price is given in basis points of 
                the notional.
            
            verbose: if False, function prints are blocked.
            
        Output:
            self.swap_price: a 1D ndarray containing the pathwise forward IRS 
                prices.
            self.mean_Monte_Carlo_price: a float specifying the mean value of 
                the pathwise swap prices.
        """
        self.fixed_rate = fixed_rate
        self.notional = notional
        self.payoff_var = payoff_var
        self.plot_timeline = plot_timeline
        self.moneyness = moneyness
        self.swap_type = swap_type
        self.tenor_structure = tenor_structure
        self.time_t = time_t
        self.units_basis_points = units_basis_points
        
        # Adjust the fixed rate if a level of moneyness was specified
        if moneyness is not None:
            self.fixed_rate = eval_moneyness_adjusted_fixed_rate(moneyness, 
                                            self.model.n_annual_trading_days, 
                                            swap_type, tenor_structure, 
                                            self.model.time_0_P_curve)
        
        self.swap_price = price_forward_start_swap(self.model.a_param, 
                                self.fixed_rate, 
                                self.model.n_annual_trading_days, 
                                self.notional, self.payoff_var, 
                                self.plot_timeline, self.model.r_t_paths, 
                                self.model.sigma, self.swap_type, 
                                self.tenor_structure, self.time_t, 
                                self.model.time_0_f_curve, 
                                self.model.time_0_P_curve, 
                                self.units_basis_points, self.model.x_t_paths, 
                                verbose)
        self.mean_Monte_Carlo_price = np.mean(self.swap_price)