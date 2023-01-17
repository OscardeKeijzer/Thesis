# -*- coding: utf-8 -*-


# Imports
from datetime import datetime
import os
import pandas as pd
import numpy as np
import scipy.stats as st
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

#%% Experiment 3: compare zero-mean process Euler simulation with antithetic 
### variables
n_annual_trading_days = 253
simulation_time = 30.
time_0_rate = .001
one_factor_HW_model = OneFactorHullWhiteModel('Simulation With Antithetic Variables Comparison', 
                                              n_annual_trading_days, 
                                              simulation_time, time_0_rate)
one_factor_HW_model.construct_curves(True, 'flat', 0.)


#%% Simulate the short rates and determine means, variances, and computation 
### times
a_param = .01
sigma = .01
n_paths = 10000
r_t_sim_time = 30.
n_repeat_simulations = 50

r_t_means_and_vars_array = np.zeros((n_repeat_simulations+2, 4))
comp_times_array = np.zeros((n_repeat_simulations+2, 2))

for count in range(n_repeat_simulations):
    # Simulate short rates without antithetic paths
    antithetic = False
    one_factor_HW_model.sim_short_rate_paths(antithetic, a_param, n_paths, 
                                             'zero-mean', r_t_sim_time, sigma, 
                                             'Euler')
    one_factor_HW_model.log_short_rate_paths()
    r_t_means_and_vars_array[count,0] = one_factor_HW_model.r_t_mean
    r_t_means_and_vars_array[count,1] = one_factor_HW_model.r_t_var
    comp_times_array[count,0] = one_factor_HW_model.short_rate_sim_runtime
    if count == n_repeat_simulations - 1:
        one_factor_HW_model.plot_short_rate_paths(None, 'both')
        
    # Simulate short rates with antithetic paths
    antithetic = True
    one_factor_HW_model.sim_short_rate_paths(antithetic, a_param, n_paths, 
                                             'zero-mean', r_t_sim_time, sigma, 
                                             'Euler')
    one_factor_HW_model.log_short_rate_paths()
    r_t_means_and_vars_array[count,2] = one_factor_HW_model.r_t_mean
    r_t_means_and_vars_array[count,3] = one_factor_HW_model.r_t_var
    comp_times_array[count,1] = one_factor_HW_model.short_rate_sim_runtime
    if count == n_repeat_simulations - 1:
        one_factor_HW_model.plot_short_rate_paths(None, 'both')
        
#%%
# Evaluate the mean of means and its standard error and store in the 
# last two rows accordingly
r_t_means_and_vars_array[n_repeat_simulations] = np.mean(r_t_means_and_vars_array[:n_repeat_simulations], 
                                                axis=0)
r_t_means_and_vars_array[n_repeat_simulations+1] = st.sem(r_t_means_and_vars_array[:n_repeat_simulations], 
                                                axis=0)
# Evaluate the mean of computation times and its standard errors and store in 
# the last two rows accordingly
comp_times_array[n_repeat_simulations] = np.mean(comp_times_array[:n_repeat_simulations], 
                                                axis=0)
comp_times_array[n_repeat_simulations+1] = st.sem(comp_times_array[:n_repeat_simulations], 
                                                axis=0)

# Save data to experiment directory
np.savetxt(one_factor_HW_model.data_dir+'means_and_vars.txt', 
           r_t_means_and_vars_array, delimiter=', ', fmt='%f', 
           header='means and variances\n'
           + ' non-antithetic mean, non-antithetic variance,' 
           +' antithetic mean, antithetic variance', 
           footer='second-to-last row: mean of means,' 
           + ' last row: SEMs')

np.savetxt(one_factor_HW_model.data_dir+'comp_times.txt', 
           comp_times_array, delimiter=', ', fmt='%f', 
           header='computation times\n'
           + ' non-antithetic, antithetic', 
           footer='second-to-last row: means,' 
           + ' last row: SEMs')

#%%
print(r_t_means_and_vars_array[:,2])
    
#%% Shutdown script
# os.system("shutdown /s /t 1")