# -*- coding: utf-8 -*-


# Imports
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
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

#%% Experiment 1: Simulate short rates using the different simulation methods 
### over varying time periods to compare mean and variance with expected values

# Simulate the one-factor Hull-White model
n_annual_trading_days = 253
simulation_time = 30.
time_0_rate = .001
one_factor_HW_model_2 = OneFactorHullWhiteModel('Simulation Method Comparison', n_annual_trading_days, 
                                                simulation_time, time_0_rate)
one_factor_HW_model_2.construct_curves(True, 'flat', 0.)

#%% Simulate the short rates
antithetic = False
a_param = .01
sigma = .01
n_paths = 10000
n_sim_time_points = 15
max_sim_time = 30.
r_t_sim_times = np.linspace(max_sim_time/n_sim_time_points, max_sim_time, 
                            num=n_sim_time_points)

# Initialize arrays for storing mean and variance of simulated short rates
r_t_direct_Euler_array = np.zeros((n_sim_time_points, 3))
r_t_direct_exact_array = np.zeros((n_sim_time_points, 3))
r_t_zero_mean_Euler_array = np.zeros((n_sim_time_points, 3))
r_t_zero_mean_exact_array = np.zeros((n_sim_time_points, 3))
expected_array = np.zeros((n_sim_time_points, 3))

r_t_direct_Euler_array[:,0] = r_t_sim_times
r_t_direct_exact_array[:,0] = r_t_sim_times
r_t_zero_mean_Euler_array[:,0] = r_t_sim_times
r_t_zero_mean_exact_array[:,0] = r_t_sim_times
expected_array[:,0] = r_t_sim_times

for count, r_t_sim_time in enumerate(r_t_sim_times):
    one_factor_HW_model_2.sim_short_rate_paths(antithetic, a_param, n_paths, 
                                               'direct', r_t_sim_time, sigma, 
                                               'Euler')
    one_factor_HW_model_2.log_short_rate_paths()
    r_t_direct_Euler_array[count,1] = one_factor_HW_model_2.r_t_mean
    r_t_direct_Euler_array[count,2] = one_factor_HW_model_2.r_t_var
    
    one_factor_HW_model_2.sim_short_rate_paths(antithetic, a_param, n_paths, 
                                               'direct', r_t_sim_time, sigma, 
                                               'exact')
    one_factor_HW_model_2.log_short_rate_paths()
    r_t_direct_exact_array[count,1] = one_factor_HW_model_2.r_t_mean
    r_t_direct_exact_array[count,2] = one_factor_HW_model_2.r_t_var
    
    one_factor_HW_model_2.sim_short_rate_paths(antithetic, a_param, n_paths, 
                                               'zero-mean', r_t_sim_time, 
                                               sigma, 'Euler')
    one_factor_HW_model_2.log_short_rate_paths()
    r_t_zero_mean_Euler_array[count,1] = one_factor_HW_model_2.r_t_mean
    r_t_zero_mean_Euler_array[count,2] = one_factor_HW_model_2.r_t_var
    
    one_factor_HW_model_2.sim_short_rate_paths(antithetic, a_param, n_paths, 
                                               'zero-mean', r_t_sim_time, 
                                               sigma, 'exact')
    one_factor_HW_model_2.log_short_rate_paths()
    r_t_zero_mean_exact_array[count,1] = one_factor_HW_model_2.r_t_mean
    r_t_zero_mean_exact_array[count,2] = one_factor_HW_model_2.r_t_var
    
    # Compute the exact mean and variance for the given simulation time
    expected_array[count,1] = one_factor_Hull_White_exp_Q(a_param, 
                                        one_factor_HW_model_2.init_forward_c, 
                                        np.array(time_0_rate), 
                                        n_annual_trading_days, sigma, 0, 
                                        r_t_sim_time)
    expected_array[count,2] = one_factor_Hull_White_var(a_param, sigma, 0, 
                                                        r_t_sim_time)
    
# Determine relative errors in means and variances
E_rel_direct_Euler_array = (r_t_direct_Euler_array - expected_array)/expected_array
E_rel_direct_exact_array = (r_t_direct_exact_array - expected_array)/expected_array
E_rel_zero_mean_Euler_array = (r_t_zero_mean_Euler_array - expected_array)/expected_array
E_rel_zero_mean_exact_array = (r_t_zero_mean_exact_array - expected_array)/expected_array

E_rel_direct_Euler_array[:,0] = r_t_sim_times
E_rel_direct_exact_array[:,0] = r_t_sim_times
E_rel_zero_mean_Euler_array[:,0] = r_t_sim_times
E_rel_zero_mean_exact_array[:,0] = r_t_sim_times

# Save data to experiment directory
np.savetxt(one_factor_HW_model_2.data_dir+'direct_Euler_means_and_vars.txt', 
           r_t_direct_Euler_array, delimiter=', ', 
           header='time (years), mean short rate, variance of short rates')
np.savetxt(one_factor_HW_model_2.data_dir+'direct_exact_means_and_vars.txt', 
           r_t_direct_exact_array, delimiter=', ', 
           header='time (years), mean short rate, variance of short rates')
np.savetxt(one_factor_HW_model_2.data_dir+'zero_mean_Euler_means_and_vars.txt', 
           r_t_zero_mean_Euler_array, delimiter=', ', 
           header='time (years), mean short rate, variance of short rates')
np.savetxt(one_factor_HW_model_2.data_dir+'zero_mean_exact_means_and_vars.txt', 
           r_t_zero_mean_exact_array, delimiter=', ', 
           header='time (years), mean short rate, variance of short rates')
np.savetxt(one_factor_HW_model_2.data_dir+'expected_means_and_vars.txt', 
           expected_array, delimiter=', ', 
           header='time (years), expectation, variance')
np.savetxt(one_factor_HW_model_2.data_dir+'direct_Euler_rel_errors.txt', 
           E_rel_direct_Euler_array, delimiter=', ', 
           header='time (years), relative error in mean, relative error in variance')
np.savetxt(one_factor_HW_model_2.data_dir+'direct_exact_rel_errors.txt', 
           E_rel_direct_exact_array, delimiter=', ', 
           header='time (years), relative error in mean, relative error in variance')
np.savetxt(one_factor_HW_model_2.data_dir+'zero_mean_Euler_rel_errors.txt', 
           E_rel_zero_mean_Euler_array, delimiter=', ', 
           header='time (years), relative error in mean, relative error in variance')
np.savetxt(one_factor_HW_model_2.data_dir+'zero_mean_exact_rel_errors.txt', 
           E_rel_zero_mean_exact_array, delimiter=', ', 
           header='time (years), relative error in mean, relative error in variance')

#%% Read in the previously saved data
r_t_direct_Euler_array = np.loadtxt(one_factor_HW_model_2.data_dir
                                    +'direct_Euler_means_and_vars.txt', 
                                    skiprows=1, delimiter=',')
r_t_direct_exact_array = np.loadtxt(one_factor_HW_model_2.data_dir
                                    +'direct_exact_means_and_vars.txt', 
                                    skiprows=1, delimiter=',')
r_t_zero_mean_Euler_array = np.loadtxt(one_factor_HW_model_2.data_dir
                                    +'zero_mean_Euler_means_and_vars.txt', 
                                    skiprows=1, delimiter=',')
r_t_zero_mean_exact_array = np.loadtxt(one_factor_HW_model_2.data_dir
                                    +'zero_mean_exact_means_and_vars.txt', 
                                    skiprows=1, delimiter=',')
expected_array = np.loadtxt(one_factor_HW_model_2.data_dir
                            +'expected_means_and_vars.txt', 
                            skiprows=1, delimiter=',')

E_rel_direct_Euler_array = np.loadtxt(one_factor_HW_model_2.data_dir
                                    +'direct_Euler_rel_errors.txt', 
                                    skiprows=1, delimiter=',')
E_rel_direct_exact_array = np.loadtxt(one_factor_HW_model_2.data_dir
                                    +'direct_exact_rel_errors.txt', 
                                    skiprows=1, delimiter=',')
E_rel_zero_mean_Euler_array = np.loadtxt(one_factor_HW_model_2.data_dir
                                    +'zero_mean_Euler_rel_errors.txt', 
                                    skiprows=1, delimiter=',')
E_rel_zero_mean_exact_array = np.loadtxt(one_factor_HW_model_2.data_dir
                                    +'zero_mean_exact_rel_errors.txt', 
                                    skiprows=1, delimiter=',')

#%% Plots
n_sim_time_points = 15
max_sim_time = 30.
r_t_sim_times = np.linspace(max_sim_time/n_sim_time_points, max_sim_time, 
                            num=n_sim_time_points)

# Plot means vs expectation
mean_plot, ax = plt.subplots(2,2, figsize=(9, 6))
mean_plot.suptitle('Mean Simulated Short Rate Values')
mean_plot.supxlabel('Simulation Time $t$ (years)')
mean_plot.supylabel(r'Mean Short Rate Value $\bar{r}_t$')

ax[0,0].plot(r_t_sim_times, r_t_direct_Euler_array[:,1], linewidth=1, 
             label='Simulation')
ax[0,0].plot(r_t_sim_times, expected_array[:,1], linewidth=1, 
            linestyle='--', label='Expectation')
ax[0,0].set_title('Direct, Euler method')
ax[0,1].plot(r_t_sim_times, r_t_direct_exact_array[:,1], linewidth=1, 
             label='Simulation')
ax[0,1].plot(r_t_sim_times, expected_array[:,1], linewidth=1, 
             linestyle='--', label='Expectation')
ax[0,1].set_title('Direct, exact simulation')
ax[1,0].plot(r_t_sim_times, r_t_zero_mean_Euler_array[:,1], linewidth=1, 
             label='Simulation')
ax[1,0].plot(r_t_sim_times, expected_array[:,1], linewidth=1, 
             linestyle='--', label='Expectation')
ax[1,0].set_title('Shifted zero-mean, Euler method')
ax[1,1].plot(r_t_sim_times, r_t_zero_mean_exact_array[:,1], linewidth=1, 
             label='Simulation')
ax[1,1].plot(r_t_sim_times, expected_array[:,1], linewidth=1, 
             linestyle='--', label='Expectation')
ax[1,1].set_title('Shifted zero-mean, exact simulation')

plt.legend(loc='best')
plt.tight_layout()
plt.savefig(one_factor_HW_model_2.figures_dir+'sim_methods_mean_comparison_plot', )

# Plot variances vs analytical variance
var_plot, ax = plt.subplots(2,2, figsize=(9, 6))
var_plot.suptitle('Variance of Simulated Short Rate Values')
var_plot.supxlabel('Simulation Time $t$ (years)')
var_plot.supylabel(r'Variance of Short Rate Value')

ax[0,0].plot(r_t_sim_times, r_t_direct_Euler_array[:,2], linewidth=1, 
             label='Simulation')
ax[0,0].plot(r_t_sim_times, expected_array[:,2], linewidth=1, 
             linestyle='--', label='Expectation')
ax[0,0].set_title('Direct, Euler method')
ax[0,1].plot(r_t_sim_times, r_t_direct_exact_array[:,2], linewidth=1, 
             label='Simulation')
ax[0,1].plot(r_t_sim_times, expected_array[:,2], linewidth=1, 
             linestyle='--', label='Expectation')
ax[0,1].set_title('Direct, exact simulation')
ax[1,0].plot(r_t_sim_times, r_t_zero_mean_Euler_array[:,2], linewidth=1, 
             label='Simulation')
ax[1,0].plot(r_t_sim_times, expected_array[:,2], linewidth=1, 
             linestyle='--', label='Expectation')
ax[1,0].set_title('Shifted zero-mean, Euler method')
ax[1,1].plot(r_t_sim_times, r_t_zero_mean_exact_array[:,2], linewidth=1, 
             label='Simulation')
ax[1,1].plot(r_t_sim_times, expected_array[:,2], linewidth=1, 
             linestyle='--', label='Expectation')
ax[1,1].set_title('Shifted zero-mean, exact simulation')

plt.legend(loc='best')
plt.tight_layout()
plt.savefig(one_factor_HW_model_2.figures_dir+'sim_methods_var_comparison_plot', )

#%% Tabulate results

# Means and relative errors of direct simulation methods
direct_methods_means_table = np.zeros((n_sim_time_points, 6))
direct_methods_means_table[:,0] = r_t_sim_times
direct_methods_means_table[:,1] = expected_array[:,1]
direct_methods_means_table[:,2] = r_t_direct_Euler_array[:,1]
direct_methods_means_table[:,3] = E_rel_direct_Euler_array[:,1]
direct_methods_means_table[:,4] = r_t_direct_exact_array[:,1]
direct_methods_means_table[:,5] = E_rel_direct_exact_array[:,1]

# Means and relative errors of shifted zero-mean simulation methods
zero_mean_methods_means_table = np.zeros((n_sim_time_points, 6))
zero_mean_methods_means_table[:,0] = r_t_sim_times
zero_mean_methods_means_table[:,1] = expected_array[:,1]
zero_mean_methods_means_table[:,2] = r_t_zero_mean_Euler_array[:,1]
zero_mean_methods_means_table[:,3] = E_rel_zero_mean_Euler_array[:,1]
zero_mean_methods_means_table[:,4] = r_t_zero_mean_exact_array[:,1]
zero_mean_methods_means_table[:,5] = E_rel_zero_mean_exact_array[:,1]

# Variances and relative errors of direct simulation methods
direct_methods_vars_table = np.zeros((n_sim_time_points, 6))
direct_methods_vars_table[:,0] = r_t_sim_times
direct_methods_vars_table[:,1] = expected_array[:,2]
direct_methods_vars_table[:,2] = r_t_direct_Euler_array[:,2]
direct_methods_vars_table[:,3] = E_rel_direct_Euler_array[:,2]
direct_methods_vars_table[:,4] = r_t_direct_exact_array[:,2]
direct_methods_vars_table[:,5] = E_rel_direct_exact_array[:,2]

# Variances and relative errors of shifted zero-mean simulation methods
zero_mean_methods_vars_table = np.zeros((n_sim_time_points, 6))
zero_mean_methods_vars_table[:,0] = r_t_sim_times
zero_mean_methods_vars_table[:,1] = expected_array[:,2]
zero_mean_methods_vars_table[:,2] = r_t_zero_mean_Euler_array[:,2]
zero_mean_methods_vars_table[:,3] = E_rel_zero_mean_Euler_array[:,2]
zero_mean_methods_vars_table[:,4] = r_t_zero_mean_exact_array[:,2]
zero_mean_methods_vars_table[:,5] = E_rel_zero_mean_exact_array[:,2]

# Print to console to copy-and-paste results into Overleaf rounded to 8 
# significant figures
print('\nMeans of direct simulation methods:')
print(tabulate(direct_methods_means_table, tablefmt="latex", floatfmt=".7E"))
print('\nMeans of shifted zero-mean simulation methods:')
print(tabulate(zero_mean_methods_means_table, tablefmt="latex", floatfmt=".7E"))
print('\nVariances of direct simulation methods:')
print(tabulate(direct_methods_vars_table, tablefmt="latex", floatfmt=".7E"))
print('\nVariances of shifted zero-mean simulation methods:')
print(tabulate(zero_mean_methods_vars_table, tablefmt="latex", floatfmt=".7E"))

#%% Shutdown script
# os.system("shutdown /s /t 1")