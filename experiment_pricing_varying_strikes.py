# -*- coding: utf-8 -*-

# Imports
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from tabulate import tabulate
import tensorflow.keras as keras

os.chdir(os.getcwd())

# Local imports
from bonds_and_bond_options import (construct_zero_coupon_bonds_curve, 
                                    price_coupon_bearing_bond, 
                                    price_coupon_bearing_bond_option, 
                                    price_zero_coupon_bond, 
                                    price_zero_coupon_bond_option_exact)
from classes import (BermudanSwaptionLSM, 
                     BermudanSwaptionRLNN, 
                     EuropeanSwaption, 
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
from neural_networks import ShallowFeedForwardNeuralNetwork
from one_factor_Hull_White_model import (gen_one_factor_Hull_White_paths)
from plotting_functions import (plot_time_series, 
                                plot_one_factor_Hull_White_histogram)
from swaps import (eval_annuity_terms, 
                   eval_swap_rate, 
                   price_forward_start_swap)
from swaptions import (most_expensive_European, 
                       price_European_swaption_exact_Jamshidian, 
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
#%% Using class for 1F-HW model:
n_annual_trading_days = 253
simulation_time = 30.
time_0_rate = .001
one_factor_HW_model = OneFactorHullWhiteModel('Pricing With Varying Strikes', 
                                                n_annual_trading_days, 
                                                simulation_time, time_0_rate)
one_factor_HW_model.construct_curves(True, 'flat', 0.)
#%% # Simulate short rates
a_param = .01
n_paths = 10000
r_t_sim_time = 30.
sigma = .01
one_factor_HW_model.sim_short_rate_paths(False, a_param, n_paths, 'zero-mean', 
                                           r_t_sim_time, sigma, 'euler', 
                                           time_0_rate)

#%% Experiment: swaption prices for varying strikes
degree = 2
fixed_rates = [.0001, .001, .01, .1]
n_fixed_rates = len(fixed_rates)
notional = 1.
payoff_var = 'swap'
plot = False
plot_regression = False
plot_timeline = False
regression_series = 'Laguerre'
swap_rate_moneyness = None
swap_type = swaption_type = 'receiver'
tenor_structure = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.]
time_t = 0.
verbose = False

# Initialize classes for the swap, European swaption, and Bermudan LSM swaption
IRS = Swap(one_factor_HW_model)
European_swaption = EuropeanSwaption(one_factor_HW_model)
Bermudan_swaption_LSM = BermudanSwaptionLSM(one_factor_HW_model)

# Initialize arrays for storing the prices of the swaps and swaptions as well 
# as the MEE gaps
swap_prices = np.zeros(n_fixed_rates)
European_exact_prices = np.zeros(n_fixed_rates)
European_MC_prices = np.zeros(n_fixed_rates)
European_MC_errors = np.zeros(n_fixed_rates)
LSM_prices = np.zeros(n_fixed_rates)
LSM_errors = np.zeros(n_fixed_rates)
RLNN_prices = np.zeros(n_fixed_rates)
RLNN_errors = np.zeros(n_fixed_rates)
LSM_MEE_gaps = np.zeros(n_fixed_rates)
RLNN_MEE_gaps = np.zeros(n_fixed_rates)

for count, fixed_rate in enumerate(fixed_rates):
    # Evaluate the swap and swaption prices
    IRS.price_forward_start_swap(fixed_rate, notional, payoff_var, False, 
                                 swap_rate_moneyness, swap_type, tenor_structure, 
                                 time_t, verbose)
    European_swaption.price_European_swaption_MC_Q(fixed_rate, notional, 
                                                   payoff_var, plot_timeline, 
                                                   swaption_type, 
                                                   swap_rate_moneyness, 
                                                   tenor_structure, time_t, 
                                                   verbose)
    European_swaption.price_European_swaption_exact_Jamshidian(a_param, 
                                                    fixed_rate, notional, 
                                                    plot_timeline, sigma, 
                                                    swaption_type, 
                                                    tenor_structure, time_t)
    Bermudan_swaption_LSM.price_Bermudan_swaption_LSM_Q(degree, fixed_rate, 
                                                        notional, payoff_var, 
                                                        plot, plot_regression, 
                                                        plot_timeline, 
                                                        regression_series, 
                                                        swap_rate_moneyness, 
                                                        swaption_type, 
                                                        tenor_structure, time_t)
    
    # Evaluate the MEE gaps
    Bermudan_swaption_LSM.eval_most_expensive_European()
    
    # Store the prices, errors, and MEE gaps in their arrays
    swap_prices[count] = IRS.swap_price
    European_exact_prices[count] = European_swaption.exact_price
    European_MC_prices[count] = European_swaption.mean_Monte_Carlo_price
    European_MC_errors[count] = European_swaption.standard_error_price
    LSM_prices[count] = Bermudan_swaption_LSM.mean_LSM_price
    LSM_errors[count] = Bermudan_swaption_LSM.standard_error_price
    LSM_MEE_gaps[count] = Bermudan_swaption_LSM.MEE_gap
    # RLNN_prices[count] = 
    # RLNN_errors[count] = 
    # RLNN_MEE_gaps[count] = 
#%% Tabulate results

output_table1 = pd.DataFrame(fixed_rates, columns=['Fixed Rate'])
output_table1['Swap Price'] = swap_prices
output_table1['European Swaption\\Price (Jamshidian)'] = European_exact_prices
output_table1['European Swaption\\Price (Monte Carlo)'] = European_MC_prices
output_table1['European Swaption\\SEM (Monte Carlo)'] = European_MC_errors

output_table2 = pd.DataFrame(fixed_rates, columns=['Fixed Rate'])
output_table2['Bermudan Swaption\\Price (LSM)'] = LSM_prices
output_table2['Bermudan Swaption\\SEM (LSM)'] = LSM_errors
output_table2['Bermudan Swaption\\MEE Gap (LSM)'] = LSM_MEE_gaps
output_table2['Bermudan Swaption\\Price (RLNN)'] = RLNN_prices
output_table2['Bermudan Swaption\\SEM (RLNN)'] = RLNN_errors
output_table2['Bermudan Swaption\\MEE Gap (RLNN)'] = RLNN_MEE_gaps
print(output_table1.to_latex(float_format="%.7f", index=False))
print(output_table2.to_latex(float_format="%.7f", index=False))
#%% RLNN: ADD WHEN IMPLEMENTATION COMPLETE!!!
#%% Read input files as e.g.:
    # input_short_rates_100000 = pd.read_csv(input_dir+'folder\\filename.extension', header=None).to_numpy()
    
# input_short_rates_100000 = read_data(input_dir
#                                        +'100,000 1F-HW Paths\\Data\\1F-HW_Short_Rate_Paths-2022-03-18_073736.parquet')
# print(input_short_rates_100000.shape)

#%% Shutdown script
# os.system("shutdown /s /t 1")