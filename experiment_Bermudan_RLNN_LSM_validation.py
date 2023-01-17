# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-

# Imports
import copy
from datetime import datetime
import keras_tuner
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy.stats as st
import tensorflow.keras as keras

os.chdir(os.getcwd())

# Local imports
from bonds_and_bond_options import (construct_zero_coupon_bonds_curve, 
                                    price_coupon_bearing_bond, 
                                    price_coupon_bearing_bond_option, 
                                    price_zero_coupon_bond, 
                                    price_zero_coupon_bond_option_exact, 
                                    price_zero_coupon_bond_option_MC)
from classes import (BermudanSwaptionLSM, 
                     BermudanSwaptionRLNN, 
                     EuropeanSwaption, 
                     EuropeanSwaptionRLNN, 
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
from neural_networks import (HyperParameterTunerShallowFeedForwardNeuralNetwork, 
                             ShallowFeedForwardNeuralNetwork)
from one_factor_Hull_White_model import (eval_discount_factors,
                                         gen_one_factor_Hull_White_paths)
from plotting_functions import (plot_time_series, 
                                plot_one_factor_Hull_White_histogram)
from swaps import (eval_annuity_terms, 
                   eval_swap_rate, 
                   price_forward_start_swap)
from swaptions import (price_European_swaption_exact_Jamshidian, 
                       price_European_swaption_MC_Q)
import tensorflow.keras as keras


#%% Initialize 1F-HW model instance:
exp_dir_name = 'Bermudan Swaption RLNN and LSM Validation Repeat for MSEs_TESTTTTTTT'
n_annual_trading_days = 253
simulation_time = 30.
time_0_rate = .03
one_factor_HW_model = OneFactorHullWhiteModel(exp_dir_name, n_annual_trading_days, 
                                              simulation_time, time_0_rate)
one_factor_HW_model.construct_curves(True, 'flat', 'continuously-compounded', 
                                     0., False)

#%%
# Short rate simulation params
a_param = .01
antithetic = True
n_paths = 12000
n_sims = 10
r_sim_time = 11.
sigma = .01

# Swaption params
fixed_rate = time_0_rate
notional = 100.
payoff_var = 'swap'
plot = False
plot_timeline = False
swap_rate_moneyness = [.8, 1., 1.2]
n_moneynesses = len(swap_rate_moneyness)
swaption_type = 'receiver'
tenor_structures = [np.arange(1., 6.+1), 
                    np.arange(3., 10.+1), 
                    np.arange(1., 11.+1)]
n_tenor_structures = len(tenor_structures)
time_t = 0.
units_basis_points = False
verbose = False

# LSM params
degree = 2
plot_regression = False
regression_series = 'power'

# RLNN params
learning_rate = .0003
n_epochs = 4500
n_hidden_nodes = 64
train_frac = 2000/n_paths

# Reference values
RLNN_ref_prices = {'1Yx5Y': {.8: 1.527, 1.: 2.543, 1.2: 4.015}, 
                   '3Yx7Y': {.8: 3.296, 1.: 4.767, 1.2: 6.625},
                   '1Yx10Y': {.8: 3.950, 1.: 5.818, 1.2: 8.346}}
LSM_ref_prices = {'1Yx5Y': {.8: 1.521, 1.: 2.534, 1.2: 4.016}, 
                  '3Yx7Y': {.8: 3.293, 1.: 4.755, 1.2: 6.629},
                  '1Yx10Y': {.8: 3.945, 1.: 5.811, 1.2: 8.353}}

# Result dictionaries
RLNN_price_dict = {'1Yx5Y': {.8: np.zeros(n_sims), 1.: np.zeros(n_sims), 1.2: np.zeros(n_sims)}, 
                   '3Yx7Y': {.8: np.zeros(n_sims), 1.: np.zeros(n_sims), 1.2: np.zeros(n_sims)},
                   '1Yx10Y': {.8: np.zeros(n_sims), 1.: np.zeros(n_sims), 1.2: np.zeros(n_sims)}}
RLNN_SE_dict = copy.deepcopy(RLNN_price_dict)
RLNN_abs_error_dict = copy.deepcopy(RLNN_price_dict)
RLNN_MSE_dict = copy.deepcopy(RLNN_price_dict)
RLNN_train_timing_dict = copy.deepcopy(RLNN_price_dict)
RLNN_price_timing_dict = copy.deepcopy(RLNN_price_dict)

LSM_price_dict = copy.deepcopy(RLNN_price_dict)
LSM_SE_dict = copy.deepcopy(RLNN_price_dict)
LSM_abs_error_dict = copy.deepcopy(RLNN_price_dict)
LSM_timing_dict = copy.deepcopy(RLNN_price_dict)

for count in range(n_sims):
    # Simulate short rates
    one_factor_HW_model.sim_short_rate_paths(a_param, antithetic, 
                                             one_factor_HW_model.init_forward_c, 
                                             one_factor_HW_model.init_forward_c_dfdT, 
                                             n_paths, None, 'zero-mean', 
                                             r_sim_time, sigma, 'Euler', 
                                             time_0_rate)
    
    for moneyness in swap_rate_moneyness:
        
        for tenor_structure in tenor_structures:
    
            # RLNN
            Bermudan_swaption_RLNN = BermudanSwaptionRLNN(fixed_rate, 
                                                one_factor_HW_model, notional, 
                                                moneyness, 
                                                swaption_type, tenor_structure, 
                                                units_basis_points)
            RLNN_train_timing_start = datetime.now()
            Bermudan_swaption_RLNN.replicate(None, time_t, 
                                             learn_rate=learning_rate, 
                                             n_epochs=n_epochs, 
                                             n_hidden_nodes=n_hidden_nodes, 
                                             train_size=train_frac)
            RLNN_train_timing_end = datetime.now()
            RLNN_price_timing_start = datetime.now()
            Bermudan_swaption_RLNN.price_direct_estimator(time_t)
            RLNN_price_timing_end = datetime.now()
            
            # LSM
            Bermudan_swaption_LSM = BermudanSwaptionLSM(one_factor_HW_model)
            Bermudan_swaption_LSM.r_t_paths = Bermudan_swaption_LSM.r_t_paths[:,Bermudan_swaption_RLNN.x_test_idxs]
            Bermudan_swaption_LSM.x_t_paths = Bermudan_swaption_LSM.x_t_paths[:,Bermudan_swaption_RLNN.x_test_idxs]
            LSM_timing_start = datetime.now()
            Bermudan_swaption_LSM.price_Bermudan_swaption_LSM_Q(degree, 
                                            fixed_rate, notional, payoff_var, 
                                            plot, plot_regression, 
                                            plot_timeline, regression_series, 
                                            moneyness, swaption_type, 
                                            tenor_structure, time_t, 
                                            units_basis_points, verbose)
            LSM_timing_end = datetime.now()
            
            # Store results
            tenor_notation = Bermudan_swaption_RLNN.tenor_structure_notation
            RLNN_price_dict[tenor_notation][moneyness][count] = Bermudan_swaption_RLNN.mean_direct_price_estimator
            RLNN_SE_dict[tenor_notation][moneyness][count] = Bermudan_swaption_RLNN.se_direct_price_estimator
            RLNN_abs_error_dict[tenor_notation][moneyness][count] = (Bermudan_swaption_RLNN.mean_direct_price_estimator 
                                                                     - RLNN_ref_prices[tenor_notation][moneyness])
            RLNN_MSE_dict[tenor_notation][moneyness][count] = Bermudan_swaption_RLNN.MSE[0]
            RLNN_train_timing_dict[tenor_notation][moneyness][count] = (RLNN_train_timing_end 
                                                                        - RLNN_train_timing_start).total_seconds()
            RLNN_price_timing_dict[tenor_notation][moneyness][count] = (RLNN_price_timing_end 
                                                                        - RLNN_price_timing_start).total_seconds()
            
            
            
            LSM_price_dict[tenor_notation][moneyness][count] = Bermudan_swaption_LSM.mean_LSM_price
            LSM_SE_dict[tenor_notation][moneyness][count] = Bermudan_swaption_LSM.se_LSM_price
            LSM_abs_error_dict[tenor_notation][moneyness][count] = (Bermudan_swaption_LSM.mean_LSM_price 
                                                                - LSM_ref_prices[tenor_notation][moneyness])
            LSM_timing_dict[tenor_notation][moneyness][count] = (LSM_timing_end 
                                                                 - LSM_timing_start).total_seconds()
            
            # Write results
            with open(one_factor_HW_model.data_dir + Bermudan_swaption_RLNN.tenor_structure_notation + f'_moneyness_{moneyness}_' + 'RLNN_LSM_validation_results.txt', 
                      'w') as f:
                f.write('\n' + 10*'*' + f' Simulation {count+1} ' + 10*'*')
                f.write(f'\nRLNN_price_dict: {RLNN_price_dict[tenor_notation][moneyness]}\n'
                        + f'RLNN_price_dict mean and SEM so far: {np.mean(RLNN_price_dict[tenor_notation][moneyness][RLNN_price_dict[tenor_notation][moneyness]!=0.]), st.sem(RLNN_price_dict[tenor_notation][moneyness][RLNN_price_dict[tenor_notation][moneyness]!=0])}\n\n'
                        + f'RLNN_SE_dict: {RLNN_SE_dict[tenor_notation][moneyness]}\n'
                        + f'RLNN_SE_dict mean and SEM so far: {np.mean(RLNN_SE_dict[tenor_notation][moneyness][RLNN_SE_dict[tenor_notation][moneyness]!=0.]), st.sem(RLNN_SE_dict[tenor_notation][moneyness][RLNN_SE_dict[tenor_notation][moneyness]!=0.])}\n\n'
                        + f'RLNN_abs_error_dict: {RLNN_abs_error_dict[tenor_notation][moneyness]}\n'
                        + f'RLNN_abs_error_dict mean and SEM so far: {np.mean(RLNN_abs_error_dict[tenor_notation][moneyness][RLNN_abs_error_dict[tenor_notation][moneyness]!=0.]), st.sem(RLNN_abs_error_dict[tenor_notation][moneyness][RLNN_abs_error_dict[tenor_notation][moneyness]!=0.])}\n\n'
                        + f'RLNN_MSE_dict: {RLNN_MSE_dict[tenor_notation][moneyness]}'
                        + f'RLNN_MSE_dict mean and SEM so far: {np.mean(RLNN_MSE_dict[tenor_notation][moneyness][RLNN_MSE_dict[tenor_notation][moneyness]!=0.]), st.sem(RLNN_MSE_dict[tenor_notation][moneyness][RLNN_MSE_dict[tenor_notation][moneyness]!=0.])}\n\n'
                        + f'RLNN_train_timing_dict: {RLNN_train_timing_dict[tenor_notation][moneyness]}\n'
                        + f'RLNN_train_timing_dict mean and SEM so far: {np.mean(RLNN_train_timing_dict[tenor_notation][moneyness][RLNN_train_timing_dict[tenor_notation][moneyness]!=0.]), st.sem(RLNN_train_timing_dict[tenor_notation][moneyness][RLNN_train_timing_dict[tenor_notation][moneyness]!=0.])}\n\n'
                        + f'RLNN_price_timing_dict: {RLNN_price_timing_dict[tenor_notation][moneyness]}\n'
                        + f'RLNN_price_timing_dict mean and SEM so far: {np.mean(RLNN_price_timing_dict[tenor_notation][moneyness][RLNN_price_timing_dict[tenor_notation][moneyness]!=0.]), st.sem(RLNN_price_timing_dict[tenor_notation][moneyness][RLNN_price_timing_dict[tenor_notation][moneyness]!=0])}\n\n\n\n'
                        + f'LSM_price_dict: {LSM_price_dict[tenor_notation][moneyness]}\n'
                        + f'LSM_price_dict mean and SEM so far: {np.mean(LSM_price_dict[tenor_notation][moneyness][LSM_price_dict[tenor_notation][moneyness]!=0.]), st.sem(LSM_price_dict[tenor_notation][moneyness][LSM_price_dict[tenor_notation][moneyness]!=0.])}\n\n'
                        + f'LSM_SE_dict: {LSM_SE_dict[tenor_notation][moneyness]}\n'
                        + f'LSM_SE_dict mean and SEM so far: {np.mean(LSM_SE_dict[tenor_notation][moneyness][LSM_SE_dict[tenor_notation][moneyness]!=0.]), st.sem(LSM_SE_dict[tenor_notation][moneyness][LSM_SE_dict[tenor_notation][moneyness]!=0.])}\n\n'
                        + f'LSM_abs_error_dict: {LSM_abs_error_dict[tenor_notation][moneyness]}\n'
                        + f'LSM_abs_error_dict mean and SEM so far: {np.mean(LSM_abs_error_dict[tenor_notation][moneyness][LSM_abs_error_dict[tenor_notation][moneyness]!=0.]), st.sem(LSM_abs_error_dict[tenor_notation][moneyness][LSM_abs_error_dict[tenor_notation][moneyness]!=0.])}\n\n'
                        + f'LSM_timing_dict: {LSM_timing_dict[tenor_notation][moneyness]}\n'
                        + f'LSM_timing_dict mean and SEM so far: {np.mean(LSM_timing_dict[tenor_notation][moneyness][LSM_timing_dict[tenor_notation][moneyness]!=0]), st.sem(LSM_timing_dict[tenor_notation][moneyness][LSM_timing_dict[tenor_notation][moneyness]!=0])}\n\n\n\n'
                        )


#%% Save numpy arrays in text files
# np.savetxt(one_factor_HW_model.data_dir+'RLNN_price_dict.txt', 
#            RLNN_price_dict, delimiter=', ', fmt='%f')
# np.savetxt(one_factor_HW_model.data_dir+'RLNN_SE_dict.txt', 
#            RLNN_SE_dict, delimiter=', ', fmt='%f')
# np.savetxt(one_factor_HW_model.data_dir+'RLNN_abs_error_dict.txt', 
#            RLNN_abs_error_dict, delimiter=', ', fmt='%f')
# np.savetxt(one_factor_HW_model.data_dir+'RLNN_train_timing_dict.txt', 
#            RLNN_train_timing_dict, delimiter=', ', fmt='%f')
# np.savetxt(one_factor_HW_model.data_dir+'RLNN_price_timing_dict.txt', 
#            RLNN_price_timing_dict, delimiter=', ', fmt='%f')

# np.savetxt(one_factor_HW_model.data_dir+'LSM_price_dict.txt', 
#            LSM_price_dict, delimiter=', ', fmt='%f')
# np.savetxt(one_factor_HW_model.data_dir+'LSM_SE_dict.txt', 
#            LSM_SE_dict, delimiter=', ', fmt='%f')
# np.savetxt(one_factor_HW_model.data_dir+'LSM_abs_error_dict.txt', 
#            LSM_abs_error_dict, delimiter=', ', fmt='%f')
# np.savetxt(one_factor_HW_model.data_dir+'LSM_timing_dict.txt', 
#            LSM_timing_dict, delimiter=', ', fmt='%f')

#%% Bar charts
fig, ax = plt.subplots(1, 1, figsize=[6, 4], sharex=True, sharey='row')
fig.supylabel('basis points of the notional' if units_basis_points 
              else 'absolute units')
fig.supxlabel('Moneyness')
x_axis_80 = np.array((0.5, 1, 1.5))
x_axis_100 = np.array((2.5, 3, 3.5))
x_axis_120 = np.array((4.5, 5, 5.5))
ax.set_xticks((1, 3, 5))
tenor_structure_labels = (r'$1Y\times5Y$', r'$3Y\times7Y$', r'$5Y\times10Y$')
xtick_labels = ('80%', '100%', '120%')
colors = ('C0', 'C1', 'C2')
ax.set_xticklabels((xtick_labels))
fig.suptitle('Absolute Errors of RLNN Bermudan Swaption Price Estimates\nWith Respect to LSM Estimates')
width = 0.5 # width of bars

units_basis_points = True
adjust_units = 10000/notional if units_basis_points else 1.

abs_errors_80 = np.array((1.5211440727661014-1.522466889565016, 
                  3.290330566126837-3.2968128224668782, 
                  3.9487659452073975-3.962845968691878))
abs_errors_100 = np.array((2.533833954781635-2.5384267421058633,
                   4.755309019652216-4.763319778029504,
                   5.803333413283887-5.805206094039691))
abs_errors_120 = np.array((4.008888548902318-4.009780290126766,
                   6.627623010112257-6.6317059661015545,
                   8.346367386742793-8.357082364446118))

# Bar chart of absolute errors
for count, tenor_structure_label in enumerate(tenor_structure_labels):
    ax.bar(x_axis_80[count], abs_errors_80[count]*adjust_units, #yerr=10*errors_LSM_T1_sigma20, 
                capsize=5, width=width, label=tenor_structure_label if count < 3 else None, 
                color=colors[count])
    ax.bar(x_axis_100[count] - 0*width, abs_errors_100[count]*adjust_units, 
                capsize=5,#yerr=10*errors_reproduction_T1_sigma20, capsize=5, 
                width=width, 
                color=colors[count])
    ax.bar(x_axis_120[count] + 0*width, abs_errors_120[count]*adjust_units, 
           width)

plt.legend(loc='best')
plt.tight_layout()

#%% 
## SHUTDOWN COMMAND!! ##
# os.system("shutdown /s /t 1")