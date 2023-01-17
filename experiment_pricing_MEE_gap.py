# -*- coding: utf-8 -*-

# Imports
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
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
one_factor_HW_model = OneFactorHullWhiteModel('MEE Gap', 
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
#%% Experiment: compute and plot MEE gap
degree = 2
fixed_rate = .001
notional = 1.
payoff_var = 'swap'
plot = False
plot_regression = False
plot_timeline = False
regression_series = 'Laguerre'
swap_rate_moneyness = None
swap_type = swaption_type = 'receiver'
tenor_structure = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.]
T_alpha = tenor_structure[0]
T_beta = tenor_structure[-1]
tenor = len(tenor_structure)
time_t = 0.
verbose = False

# Initialize classes for the swap, European swaption, and Bermudan swaption
European_swaption = EuropeanSwaption(one_factor_HW_model)
Bermudan_swaption_LSM = BermudanSwaptionLSM(one_factor_HW_model)

# Evaluate the Bermudan swaption price and MEE gap
Bermudan_swaption_LSM.price_Bermudan_swaption_LSM_Q(degree, fixed_rate, 
                                                    notional, payoff_var, 
                                                    plot, plot_regression, 
                                                    plot_timeline, 
                                                    regression_series, 
                                                    swap_rate_moneyness, 
                                                    swaption_type, 
                                                    tenor_structure, time_t)
Bermudan_price = Bermudan_swaption_LSM.mean_LSM_price
Bermudan_SEM = Bermudan_swaption_LSM.standard_error_price

Bermudan_swaption_LSM.eval_most_expensive_European()
MEE_gap = Bermudan_swaption_LSM.MEE_gap


# Initialize array for storing the prices of the European swaptions spanned by 
# the Bermudan
spanned_European_prices = np.zeros(tenor-1)

for count, __ in enumerate(tenor_structure[:-1]):
    European_swaption.price_European_swaption_exact_Jamshidian(a_param, 
                                                    fixed_rate, notional, 
                                                    plot_timeline, sigma, 
                                                    swaption_type, 
                                                    tenor_structure[count:], 
                                                    time_t)
    spanned_European_prices[count] = European_swaption.exact_price

#%% Plot results
MEE_plot, ax = plt.subplots(1,1)
x_axis = tenor_structure[:-1]
ax.set_xlabel(r'European Swaption $T_\alpha$')
ax.set_xticks(tenor_structure[:-1], np.array(tenor_structure[:-1]).astype(int))
ax.set_ylabel('Price')
plt.suptitle(fr'Bermudan-MEE Gap for a ${{{int(T_alpha)}}}\times{{{int(T_beta)}}}$' 
             + ' Bermudan Swaption')
plt.vlines(x=4, ymin=spanned_European_prices[3], 
           ymax=spanned_European_prices[3]+MEE_gap, 
           color='red', label='Bermudan-MEE Gap', linestyle='--', linewidth=2)
ax.plot(x_axis, np.repeat(Bermudan_price, tenor-1), color='black', linewidth=1, 
        label='Bermudan LSM Price')
ax.fill_between(x_axis, np.repeat(Bermudan_price - Bermudan_SEM, tenor-1), 
                np.repeat(Bermudan_price + Bermudan_SEM, tenor-1), 
                alpha=.5, label='Bermudan LSM Price Standard Error')
ax.plot(x_axis, spanned_European_prices, color='black', linestyle='--', 
        linewidth=1, marker='o', markersize=10, label='Spanned European Prices')
ax.legend(loc='best')
plt.tight_layout
plt.savefig(one_factor_HW_model.figures_dir+'MEE_gap_plot', 
            bbox_inches="tight")

#%% RLNN: ADD WHEN IMPLEMENTATION COMPLETE!!!
#%% Read input files as e.g.:
    # input_short_rates_100000 = pd.read_csv(input_dir+'folder\\filename.extension', header=None).to_numpy()
    
# input_short_rates_100000 = read_data(input_dir
#                                        +'100,000 1F-HW Paths\\Data\\1F-HW_Short_Rate_Paths-2022-03-18_073736.parquet')
# print(input_short_rates_100000.shape)

#%% Shutdown script
# os.system("shutdown /s /t 1")