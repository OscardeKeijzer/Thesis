# -*- coding: utf-8 -*-
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
exp_dir_name = 'Bermudan 1YX5Y Swap RLNN Pricing Final Hyperparams Latest'
n_annual_trading_days = 365
simulation_time = 30.
time_0_rate = .001
one_factor_HW_model = OneFactorHullWhiteModel(exp_dir_name, n_annual_trading_days, 
                                              simulation_time, time_0_rate)
one_factor_HW_model.construct_curves(True, 'flat', 'continuously-compounded', 
                                     0., False)

#%%
# Short rate simulation parameters
a_param = .01
antithetic = True
r_simulation_time = 6.
seed = 3
sigma = .01
n_paths = 15000
n_test = 10000

# Swaption and pricing parameters
fixed_rate = .0305
notional = 1.
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

# Vary RLNN parameters around Ajda's swap hyperparameters
orig_n_hidden_units = 64
orig_learning_rate = .0002
orig_n_epochs = 1000
orig_n_train = 5000

n_hyperparams = 4
n_adjustments = 3
n_repeats = 4

learning_rates = np.array((#np.repeat(orig_learning_rate*4, n_repeats), 
                         np.repeat(orig_learning_rate*4, n_repeats), 
                         np.repeat(orig_learning_rate, n_repeats), 
                         np.repeat(orig_learning_rate/4, n_repeats))).flatten()
ns_epochs = np.array((#np.repeat(orig_n_epochs/2, n_repeats), 
                     np.repeat(orig_n_epochs, n_repeats), 
                     np.repeat(orig_n_epochs*2, n_repeats), 
                     np.repeat(orig_n_epochs*4, n_repeats)), dtype=int).flatten()
ns_hidden_units = np.array((#np.repeat(orig_n_hidden_units/4, n_repeats), 
                           np.repeat(orig_n_hidden_units/2, n_repeats), 
                           np.repeat(orig_n_hidden_units, n_repeats), 
                           np.repeat(orig_n_hidden_units*2, n_repeats))).flatten()
ns_train = np.array((#np.repeat(orig_n_train/4, n_repeats), 
                     np.repeat(orig_n_train/2, n_repeats), 
                     np.repeat(orig_n_train, n_repeats), 
                     np.repeat(orig_n_train*2, n_repeats))).flatten()

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
timing_vector = np.zeros_like(mean_price_vector)

# Initialize original hyperparameters
learning_rate = orig_learning_rate
n_epochs = orig_n_epochs
n_hidden_units = orig_n_hidden_units
n_train = orig_n_train
train_frac = n_train/n_paths


one_factor_HW_model.sim_short_rate_paths(a_param, antithetic, 
                                         one_factor_HW_model.init_forward_c, 
                                         one_factor_HW_model.init_forward_c_dfdT, 
                                         n_paths, seed,'zero-mean', 
                                         r_simulation_time, sigma, 'Euler', 
                                         time_0_rate)

for n_sim in range(n_simulations):
    print(f'n_sim: {n_sim}')
    # Simulate short rates
    if n_sim < int(n_adjustments*n_repeats):
        learning_rate = float(learning_rates[n_sim])
    elif n_sim < int(2*n_adjustments*n_repeats):
        learning_rate = orig_learning_rate
        n_epochs = int(ns_epochs[int(n_sim%int(n_adjustments*n_repeats))])
    elif n_sim < int(3*n_adjustments*n_repeats):
        n_epochs = orig_n_epochs
        n_hidden_units = int(ns_hidden_units[int(n_sim%int(n_adjustments*n_repeats))])
    elif n_sim < int(4*n_adjustments*n_repeats):
        n_hidden_units = orig_n_hidden_units
        n_train = int(ns_train[int(n_sim%int(n_adjustments*n_repeats))])
        
    n_paths = int(n_train + n_test)
    train_frac = n_train/n_paths
    n_test_paths = int(n_paths*(1 - train_frac))
        
    if (n_sim == 3*n_adjustments*n_repeats 
        or n_sim == 3*n_adjustments*n_repeats+n_repeats 
        or n_sim == 3*n_adjustments*n_repeats+2*n_repeats 
        or n_sim == 3*n_adjustments*n_repeats+3*n_repeats):
        one_factor_HW_model.sim_short_rate_paths(a_param, antithetic, 
                                                 one_factor_HW_model.init_forward_c, 
                                                 one_factor_HW_model.init_forward_c_dfdT, 
                                                 n_paths, seed, 'zero-mean', 
                                                 r_simulation_time, sigma, 
                                                 'Euler', time_0_rate)
    
    # Price receiver swaptions with selected tenor structures
    for count, tenor_structure in enumerate(tenor_structures):
        print(f'tenor_structure: {tenor_structure}')
        
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
        
        with open(one_factor_HW_model.data_dir + 'BermudanRLNN_price_and_rel_err_results.txt', 
                  'a') as f:
            f.write(f'\nmean_price_vector:\n{mean_price_vector}\n')
            f.write(f'\nse_vector:\n{se_vector}\n')
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

np.savetxt(one_factor_HW_model.data_dir+'mean_price_vector_LSM.txt', 
           mean_price_vector[1], delimiter=', ', fmt='%f', 
           header='Bermudan Swaption Price Vector in Basis Points of the Notional')
np.savetxt(one_factor_HW_model.data_dir+'se_vector_LSM.txt', 
           se_vector[1], delimiter=', ', fmt='%f', 
           header='Bermudan Swaption Price Standard Error Vector in Basis Points of the Notional')
np.savetxt(one_factor_HW_model.data_dir+'timing_vector_LSM.txt', 
           timing_vector[1], delimiter=', ', fmt='%f', 
           header='Bermudan Swaption Price Function Timing Vector')

#%%
## SHUTDOWN COMMAND!! ##
os.system("shutdown /s /t 1")