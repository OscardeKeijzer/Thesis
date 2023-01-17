# -*- coding: utf-8 -*-

# Imports
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
exp_dir_name = 'Bermudan Swaption RLNN Hyperparameter Convergence'
n_annual_trading_days = 253
simulation_time = 6.
time_0_rate = .03
one_factor_HW_model = OneFactorHullWhiteModel(exp_dir_name, n_annual_trading_days, 
                                              simulation_time, time_0_rate)
one_factor_HW_model.construct_curves(True, 'flat', 'continuously-compounded', 
                                     0., False)

#%%
# Vary RLNN parameters around Jori's hyperparameters
orig_n_hidden_units = 64
orig_learning_rate = np.float32(.0003)
orig_n_epochs = 4500
orig_n_train = 2000

# Short rate simulation parameters
a_param = .01
antithetic = True
r_simulation_time = 6.
seed = None
sigma = .01
n_paths = orig_n_train + 10000

# Swaption and pricing parameters
fixed_rate = .0305
notional = 100.
payoff_var = 'swap'
plot = False
plot_regression = False
plot_timeline = False
swap_rate_moneyness = 1.
swap_type = swaption_type = 'receiver'
tenor_structures = [np.arange(1., 6.+1)]
time_t = 0.
units_basis_points = True
verbose = False

# LSM parameters
degree = 2
regression_series = 'power'

n_hyperparams = 4
n_adjustments = 5
n_repeats = 10

hyper_params_dict = {'n_hidden_units': np.repeat((np.concatenate((np.array((orig_n_hidden_units/8, 
                                                 orig_n_hidden_units/4, 
                                                 orig_n_hidden_units, 
                                                 orig_n_hidden_units*4, 
                                                 orig_n_hidden_units*8), 
                                                                dtype=int), 
                                        np.repeat((orig_n_hidden_units), 15)))), n_repeats), 
                     'learning_rates': np.repeat((np.concatenate((np.repeat((orig_learning_rate), 5), 
                                                       np.array((orig_learning_rate/9, 
                                                                 orig_learning_rate/3, 
                                                                 orig_learning_rate, 
                                                                 orig_learning_rate*3, 
                                                                 orig_learning_rate*9)), 
                                                       np.repeat((orig_learning_rate), 10))).astype(np.float32)), 10),
                     'n_epochs': np.repeat((np.concatenate((np.repeat((orig_n_epochs), n_repeats), 
                                                 np.array((orig_n_epochs/3, 
                                                           orig_n_epochs/2, 
                                                           orig_n_epochs, 
                                                           orig_n_epochs*2, 
                                                           orig_n_epochs*3), 
                                                          dtype=int), 
                                                 np.repeat((orig_n_epochs), 5)))), n_repeats), 
                     'n_train': np.repeat((np.concatenate((np.repeat((orig_n_train), 15), 
                                                np.array((orig_n_train/4, 
                                                          orig_n_train/2, 
                                                          orig_n_train, 
                                                          orig_n_train*2, 
                                                          orig_n_train*4), 
                                                         dtype=int)))), n_repeats)}

print(hyper_params_dict)
assert 1==2

# Other RLNN parameters
save_weights = True
seed_biases = 1
seed_weights = 2
test_fit = False

# Run pricing experiment
n_simulations = n_hyperparams*n_adjustments*n_repeats

# Initialize price, error and timing vectors. The RLNN prices are stored in the 
# first rows and the Monte Carlo function prices are stored in the second
mean_price_vector = np.zeros((2, len(tenor_structures), n_simulations))
se_vector = np.zeros_like(mean_price_vector)
mean_abs_error_vector = np.zeros((len(tenor_structures), n_simulations))
MSE_vector = np.zeros((len(tenor_structures), n_simulations))
timing_vector = np.zeros_like(mean_price_vector)

# # Initialize original hyperparameters
# learning_rate = orig_learning_rate
# n_epochs = orig_n_epochs
# n_hidden_units = orig_n_hidden_units
# n_train = orig_n_train
# train_frac = n_train/n_paths


for n_sim in range(n_simulations):
    print(f'n_sim: {n_sim}')
    n_hidden_units = hyper_params_dict['n_hidden_units'][n_sim]
    learning_rate = hyper_params_dict['learning_rates'][n_sim]
    n_epochs = hyper_params_dict['n_epochs'][n_sim]
    n_train = hyper_params_dict['n_train'][n_sim]
    n_paths = int(n_train + 10000)
    train_frac = n_train/n_paths
    
    # Simulate short rates
    one_factor_HW_model.sim_short_rate_paths(a_param, antithetic, 
                                             one_factor_HW_model.init_forward_c, 
                                             one_factor_HW_model.init_forward_c_dfdT, 
                                             n_paths, seed,'zero-mean', 
                                             r_simulation_time, sigma, 'Euler', 
                                             time_0_rate)
    
    # Price receiver swaptions with selected tenor structures
    for count, tenor_structure in enumerate(tenor_structures):
        # print(f'tenor_structure: {tenor_structure}')
        
        # Initialize RLNN object and evaluate swaption price
        RLNNBermudanSwaption = BermudanSwaptionRLNN(fixed_rate, 
                                            one_factor_HW_model, notional, 
                                            swap_rate_moneyness, swaption_type, 
                                            tenor_structure, n_sim, 
                                            units_basis_points)
        
        RLNN_starting_time = datetime.now()
        RLNNBermudanSwaption.replicate(None, time_t, learn_rate=learning_rate, 
                               n_epochs=n_epochs, n_hidden_nodes=n_hidden_units, 
                               seed_biases=seed_biases, seed_weights=seed_weights, 
                               save_weights=save_weights, test_fit=test_fit, 
                               train_size=train_frac)
        RLNNBermudanSwaption.price_direct_estimator(time_t)
        RLNN_finish_time = datetime.now()
        
        # Initialize Monte Carlo pricing object, select the test paths, and 
        # evaluate swaption price
        LSMBermudanSwaption = BermudanSwaptionLSM(one_factor_HW_model)
        LSMBermudanSwaption.r_t_paths = LSMBermudanSwaption.r_t_paths[:,RLNNBermudanSwaption.x_train_idxs]
        LSMBermudanSwaption.x_t_paths = LSMBermudanSwaption.x_t_paths[:,RLNNBermudanSwaption.x_train_idxs]
        
        MC_starting_time = datetime.now()
        LSMBermudanSwaption.price_Bermudan_swaption_LSM_Q(degree, fixed_rate, 
                                                            notional, payoff_var, 
                                                            plot, plot_regression, 
                                                            plot_timeline, 
                                                            regression_series, 
                                                            swap_rate_moneyness, 
                                                            swaption_type, 
                                                            tenor_structure, time_t, 
                                                            units_basis_points, verbose)
        MC_finish_time = datetime.now()
        
        # Store the prices and errors in vectors
        mean_price_vector[0,count,n_sim] = RLNNBermudanSwaption.mean_direct_price_estimator
        se_vector[0,count,n_sim] = RLNNBermudanSwaption.se_direct_price_estimator
        timing_vector[0,count,n_sim] = (RLNN_finish_time 
                                  - RLNN_starting_time).total_seconds()
        
        mean_price_vector[1,count,n_sim] = LSMBermudanSwaption.mean_LSM_price
        se_vector[1,count,n_sim] = LSMBermudanSwaption.se_LSM_price
        timing_vector[1,count,n_sim] = (MC_finish_time
                                  - MC_starting_time).total_seconds()
        
        mean_abs_error_vector[count,n_sim] = (RLNNBermudanSwaption.mean_direct_price_estimator 
                                 - LSMBermudanSwaption.mean_LSM_price)
        MSE_vector[count,n_sim] = RLNNBermudanSwaption.MSE[0]
        
        with open(one_factor_HW_model.data_dir + 'BermudanRLNN_price_and_rel_err_results.txt', 
                  'w') as f:
            f.write(f'\nmean_price_vector:\n{mean_price_vector}\n')
            f.write(f'\nse_vector:\n{se_vector}\n')
            f.write(f'\nmean_abs_error_vector:\n{mean_abs_error_vector}\n')
            f.write(f'\nMSE_vector:\n{MSE_vector}\n')
            f.write(f'\ntiming_vector:\n{timing_vector}\n')
            f.write(f'\n\nabs. error with respect to LMS:\n{mean_price_vector[0]-mean_price_vector[1]}')
#%%
np.savetxt(one_factor_HW_model.data_dir+'mean_price_vector_RLNN.txt', 
           mean_price_vector[0], delimiter=', ', fmt='%f', 
           header='Bermudan Swaption Price Vector in Basis Points of the Notional')
np.savetxt(one_factor_HW_model.data_dir+'se_vector_RLNN.txt', 
           se_vector[0], delimiter=', ', fmt='%f', 
           header='Bermudan Swaption Price Standard Error Vector in Basis Points of the Notional')
np.savetxt(one_factor_HW_model.data_dir+'timing_vector_RLNN.txt', 
           timing_vector[0], delimiter=', ', fmt='%f', 
           header='Bermudan Swaption Price Function Timing Vector')
np.savetxt(one_factor_HW_model.data_dir+'mean_abs_error_RLNN.txt', 
           mean_abs_error_vector, delimiter=', ', fmt='%f')
np.savetxt(one_factor_HW_model.data_dir+'MSE_vector_RLNN.txt', 
           MSE_vector, delimiter=', ', fmt='%f')

np.savetxt(one_factor_HW_model.data_dir+'mean_price_vector_LSM.txt', 
           mean_price_vector[1], delimiter=', ', fmt='%f', 
           header='Bermudan Swaption Price Vector in Basis Points of the Notional')
np.savetxt(one_factor_HW_model.data_dir+'se_vector_LSM.txt', 
           se_vector[1], delimiter=', ', fmt='%f', 
           header='Bermudan Swaption Price Standard Error Vector in Basis Points of the Notional')
np.savetxt(one_factor_HW_model.data_dir+'timing_vector_LSM.txt', 
           timing_vector[1], delimiter=', ', fmt='%f', 
           header='Bermudan Swaption Price Function Timing Vector')

#%% Plot results
fig, ax = plt.subplots(2, 2, figsize=(9,6))
fig.suptitle('RLNN Bermudan Pricing Parameter Convergence')
fig.supylabel('Abs. Err. (bp of notional)')
# n_hidden_units
x_axis = (4, 16, 64, 256, 512)
y_axis = (8.0184361881595, 8.1251872253427, 7.4755749796684, 6.8279336512435, 
          5.5113513027358)
SEMs = (2.3639843755119, 2.1591822050439, 1.7428423101981, 
        2.5694098047387, 1.2707410986629)
ax[0,0].scatter(x_axis, y_axis)
ax[0,0].errorbar(x_axis, y_axis, SEMs, fmt='o', alpha=.5)
ax[0,0].set_xscale('log', base=2)
ax[0,0].set_xticks(x_axis)
ax[0,0].set_xticklabels(x_axis)
ax[0,0].set_xlabel('# Hidden Units')

# learning_rates
x_axis = (.000033333333, .0001, .0003, .0009, .0027)
y_axis = (4.5699300364761, 5.6614868712557, 4.0300360378799, 
          5.1481103491658, 5.7052687586146)
SEMs = (1.4451387662881, 1.7903193456308, 1.2744092932261, 1.6279754349249, 
        1.8041643940623)
ax[0,1].scatter(x_axis, y_axis)
ax[0,1].errorbar(x_axis, y_axis, SEMs, fmt='o', alpha=.5)
ax[0,1].set_xscale('log', base=3)
ax[0,1].set_xticks(x_axis)
ax[0,1].set_xticklabels(x_axis)
# ax[0,1].ticklabel_format(style='sci', axis='x')
ax[0,1].set_xlabel('Learning Rate')

# n_epochs
x_axis = (1500, 2250, 4500, 9000, 13500)
y_axis = (5.1506165181501, 4.5975844850119, 4.870781753424, 
          6.4489084064192, 2.0393238986092)
SEMs = (1.628767955144, 1.453883870769, 1.5402764326408, 2.0393238986092, 
        1.7104414882496)
ax[1,0].scatter(x_axis, y_axis)
ax[1,0].errorbar(x_axis, y_axis, SEMs, fmt='o', alpha=.5)
ax[1,0].set_xscale('log')
ax[1,0].set_xticks(x_axis)
ax[1,0].set_xticklabels(x_axis)
ax[1,0].set_xlabel('# Epochs')

# n_train
x_axis = (500, 1000, 2000, 4000, 8000)
y_axis = (14.881714848335, 5.6469612248006, 7.2519181263909, 
          4.49674926756, 2.3114950838782)
SEMs = (4.7060114409886, 1.7857259329024, 1.7060114409886, 
                 1.4219969752183, 0.7309589265337)
ax[1,1].scatter(x_axis, y_axis)
ax[1,1].errorbar(x_axis, y_axis, SEMs, fmt='o', alpha=.5)
ax[1,1].set_xscale('log')
ax[1,1].set_xticks(x_axis)
ax[1,1].set_xticklabels(x_axis)
ax[1,1].set_xlabel('# Training Paths')

plt.tight_layout()
#%%
## SHUTDOWN COMMAND!! ##
# os.system("shutdown /s /t 1")