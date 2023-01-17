# -*- coding: utf-8 -*-


# Imports
from datetime import datetime
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from tabulate import tabulate
os.chdir(os.getcwd())

# Local imports
from bonds_and_bond_options import (construct_zero_coupon_bonds_curve, 
                                    price_coupon_bearing_bond, 
                                    price_coupon_bearing_bond_option, 
                                    price_zero_coupon_bond, 
                                    price_zero_coupon_bond_option_exact)
from classes import (BermudanSwaptionRLNN, 
                     OneFactorHullWhiteModel, 
                     Swap)
from curve_functions import (construct_time_0_inst_forw_rate_curve,
                             construct_time_0_inst_forw_rate_curve_derivative,
                             construct_time_0_zero_coupon_curve, 
                             construct_time_0_zero_bond_curve)
from data_functions import (read_Parquet_data, 
                            write_Parquet_data)
from least_squares_Monte_Carlo import (price_Bermudan_stock_option_LSM, 
                                       price_Bermudan_swaption_LSM_Q)
from one_factor_Hull_White_model import (gen_one_factor_Hull_White_paths,
                                         one_factor_Hull_White_exp_Q,
                                         one_factor_Hull_White_var)
from plotting_functions import (plot_time_series, 
                                plot_one_factor_Hull_White_histogram)
from swaps import (eval_annuity_terms, 
                   eval_swap_rate, 
                   price_forward_start_swap)
from swaptions import (price_European_swaption_exact_Jamshidian, 
                       price_European_swaption_MC_Q)

# Plotting style parameters
plt.style.use('fivethirtyeight')
plt.rcParams["figure.dpi"] = 150
plt.rcParams["figure.figsize"] = [6, 4]
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 14
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#%% Experiment 2: Simulate short rates using the different simulation methods 
### for a year using repeat simulations

# Simulate the one-factor Hull-White model
n_annual_trading_days = 253
simulation_time = 30.
time_0_rate = .001
one_factor_HW_model = OneFactorHullWhiteModel('Simulation Method Computation Time', 
                                                n_annual_trading_days, 
                                                simulation_time, time_0_rate)
one_factor_HW_model.construct_curves(True, 'flat', 0.)

#%% Simulate the short rates
a_param = .01
antithetic = True
sigma = .01
n_paths = 10000
r_t_sim_time = 1.
n_repeat_simulations = 50

# Initialize array for storing computation times with indices 0, 1, 2, and 3 
# corresponding to direct Euler, direct exact, zero-mean Euler, and 
# zero-mean exact, respectively
comp_time_array = np.zeros((n_repeat_simulations+2, 4))

for count in range(n_repeat_simulations):
    one_factor_HW_model.sim_short_rate_paths(antithetic, a_param, n_paths, 
                                             'direct', r_t_sim_time, sigma, 
                                             'Euler')
    one_factor_HW_model.log_short_rate_paths()
    comp_time_array[count,0] = one_factor_HW_model.short_rate_sim_runtime
    
    one_factor_HW_model.sim_short_rate_paths(antithetic, a_param, n_paths, 
                                             'direct', r_t_sim_time, sigma, 
                                             'exact')
    one_factor_HW_model.log_short_rate_paths()
    comp_time_array[count,1] = one_factor_HW_model.short_rate_sim_runtime
    
    one_factor_HW_model.sim_short_rate_paths(antithetic, a_param, n_paths, 
                                             'zero-mean', r_t_sim_time, sigma, 
                                             'Euler')
    one_factor_HW_model.log_short_rate_paths()
    comp_time_array[count,2] = one_factor_HW_model.short_rate_sim_runtime
    
    one_factor_HW_model.sim_short_rate_paths(antithetic, a_param, n_paths, 
                                             'zero-mean', r_t_sim_time, sigma, 
                                             'exact')
    one_factor_HW_model.log_short_rate_paths()
    comp_time_array[count,3] = one_factor_HW_model.short_rate_sim_runtime
    
# Evaluate the mean computation time and its standard error and store in the 
# last two rows accordingly
comp_time_array[n_repeat_simulations] = np.mean(comp_time_array[:n_repeat_simulations], 
                                                axis=0)
comp_time_array[n_repeat_simulations+1] = st.sem(comp_time_array[:n_repeat_simulations], 
                                                axis=0)

# Save data to experiment directory
np.savetxt(one_factor_HW_model.data_dir+'computation_times.txt', 
           comp_time_array, delimiter=', ', fmt='%f', 
           header='computation times (seconds)/nDirect Euler, direct exact' 
           + ' zero-mean Euler, zero-mean exact', 
           footer='second-to-last row: mean computation time,' 
           + ' last row: SEM of computation time')


#%% Read in the previously saved data
comp_time_arrayww = np.loadtxt(one_factor_HW_model.data_dir
                                    +'computation_times.txt', 
                                    skiprows=1, delimiter=',')

#%% Shutdown script
# os.system("shutdown /s /t 1")