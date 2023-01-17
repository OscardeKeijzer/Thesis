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
from classes import (EuropeanSwaption, 
                     EuropeanSwaptionRLNN, 
                     OneFactorHullWhiteModel)
from data_functions import (read_Parquet_data, 
                            write_Parquet_data)
from plotting_functions import (plot_time_series, 
                                plot_one_factor_Hull_White_histogram)


#%% Initialize 1F-HW model instance:
exp_dir_name = 'Euro Swaption RLNN Pricing Ajda Swaption Hyperparams' # 'Euro Swaption RLNN Pricing Ajda Swap Hyperparams' 
n_annual_trading_days = 253
simulation_time = 30.
time_0_rate = .001
one_factor_HW_model = OneFactorHullWhiteModel(exp_dir_name, n_annual_trading_days, 
                                              simulation_time, time_0_rate)
one_factor_HW_model.construct_curves(True, 'flat', 'continuously-compounded', 
                                     0., False)

#%% Price European swaption using Monte Carlo and exact functions
# Short rate simulation parameters
a_param = .01
antithetic = True
n_paths = 11500 # 11500
r_simulation_time = 6.
seed = None
sigma = .01

# Swaption and pricing parameters
fixed_rate = .0305
notional = 1.
payoff_var = 'swap'
plot = False
plot_regression = False
plot_timeline = False
swap_rate_moneyness = 1.
swap_type = swaption_type = 'receiver'
tenor_structures = [np.arange(1., 6.+1)]#, 
                    # np.arange(5., 15.+1), 
                    # np.arange(10., 15.+1)]

time_t = 0.
units_basis_points = False
verbose = False

# RLNN parameters Ajda swaption
n_hidden_units = 32
learning_rate = .0004
n_epochs = 2000
train_frac = 1500/n_paths

# RLNN parameters Ajda swap
# n_hidden_units = 64
# learning_rate = .0002
# n_epochs = 1000
# train_frac = 5000/n_paths

# Other RLNN parameters
save_weights = True
seed_biases = None
seed_weights = None
test_fit = False
n_test_paths = int(n_paths*(1 - train_frac))

# Run pricing experiment
n_simulations = 50

# Initialize price, error and timing vectors. The RLNN prices are stored in the 
# first rows and the Monte Carlo function prices are stored in the second
mean_price_vector = np.zeros((2, 3, n_simulations))
se_vector = np.zeros_like(mean_price_vector)
mean_abs_err_vector = np.zeros_like(mean_price_vector)
se_abs_err_vector = np.zeros_like(mean_price_vector)
timing_vector = np.zeros_like(mean_price_vector)
MSE_vector = np.zeros((3, n_simulations))

for n_sim in range(n_simulations):
    print(f'n_sim: {n_sim}')
    # Simulate short rates
    one_factor_HW_model.sim_short_rate_paths(a_param, antithetic, 
                                             one_factor_HW_model.init_forward_c, 
                                             one_factor_HW_model.init_forward_c_dfdT, 
                                             n_paths, seed,'zero-mean', 
                                             r_simulation_time, sigma, 
                                             'Euler', time_0_rate)
    
    # Price receiver swaptions with selected tenor structures
    for count, tenor_structure in enumerate(tenor_structures):
        print(f'tenor_structure: {tenor_structure}')
        
        # Initialize RLNN object and evaluate swaption price
        RLNNEuropeanSwaption = EuropeanSwaptionRLNN(fixed_rate, 
                                            one_factor_HW_model, notional, 
                                            swap_rate_moneyness, swaption_type, 
                                            tenor_structure, n_sim, 
                                            units_basis_points)
        
        RLNN_starting_time = datetime.now()
        RLNNEuropeanSwaption.replicate(None, time_t, learn_rate=learning_rate, 
                               n_epochs=n_epochs, n_hidden_nodes=n_hidden_units, 
                               seed_biases=seed_biases, seed_weights=seed_weights, 
                               save_weights=save_weights, test_fit=test_fit, 
                               train_size=train_frac)
        RLNNEuropeanSwaption.price_direct_estimator(time_t)
        RLNN_finish_time = datetime.now()
        
        # Initialize Monte Carlo pricing object, select the test paths, and 
        # evaluate swaption price
        MCEuropeanSwaption = EuropeanSwaption(fixed_rate, one_factor_HW_model, 
                                              notional, swaption_type, 
                                              swap_rate_moneyness, tenor_structure, 
                                              time_t, units_basis_points)
        MCEuropeanSwaption.r_t_paths = MCEuropeanSwaption.r_t_paths[:,RLNNEuropeanSwaption.x_train_idxs]
        MCEuropeanSwaption.x_t_paths = MCEuropeanSwaption.x_t_paths[:,RLNNEuropeanSwaption.x_train_idxs]
        
        MC_starting_time = datetime.now()
        MCEuropeanSwaption.price_European_swaption_MC_Q(payoff_var, plot_timeline, 
                                                        verbose)
        MC_finish_time = datetime.now()
        
        MCEuropeanSwaption.price_European_swaption_Bachelier(verbose)
        
        # Store the prices and errors in vectors
        mean_price_vector[0,count,n_sim] = RLNNEuropeanSwaption.mean_direct_price_estimator
        se_vector[0,count,n_sim] = RLNNEuropeanSwaption.se_direct_price_estimator
        mean_abs_err_vector[0,count,n_sim] = RLNNEuropeanSwaption.mean_abs_err_direct_price_estimator
        se_abs_err_vector[0,count,n_sim] = RLNNEuropeanSwaption.se_abs_error_direct_price_estimator
        timing_vector[0,count,n_sim] = (RLNN_finish_time 
                                  - RLNN_starting_time).total_seconds()
        MSE_vector[count,n_sim] = RLNNEuropeanSwaption.MSE
        
        mean_price_vector[1,count,n_sim] = MCEuropeanSwaption.mean_Monte_Carlo_price
        se_vector[1,count,n_sim] = MCEuropeanSwaption.se_Monte_Carlo_price
        mean_abs_err_vector[1,count,n_sim] = MCEuropeanSwaption.mean_abs_err_Monte_Carlo_price
        se_abs_err_vector[1,count,n_sim] = MCEuropeanSwaption.se_abs_err_Monte_Carlo_price
        timing_vector[1,count,n_sim] = (MC_finish_time 
                                  - MC_starting_time).total_seconds()
        
        with open(one_factor_HW_model.data_dir + 'EuropeanRLNN_price_and_abs_err_results.txt', 
                  'w') as f:
            f.write(f'\nParameters:\nn_paths: {n_paths}, n_hidden_nodes: {n_hidden_units}, ' 
                    + f'learning_rate: {learning_rate}, n_epochs: {n_epochs}, '  
                    + f'train_frac: {train_frac}' )
            f.write(f'Bachelier price: {MCEuropeanSwaption.exact_price_Bachelier}')
            f.write(f'\nmean_price_vector:\n{mean_price_vector}\n')
            f.write(f'\nse_vector:\n{se_vector}\n')
            f.write(f'\nMSE_vector:\n{MSE_vector}\n')
            f.write(f'\nmean_abs_err_vector:\n{mean_abs_err_vector}\n')
            f.write(f'\nse_abs_err_vector:\n{se_abs_err_vector}\n')
            f.write(f'\ntiming_vector:\n{timing_vector}\n\n')
#%%
np.savetxt(one_factor_HW_model.data_dir+'mean_price_vector_RLNN.txt', 
           mean_price_vector[0], delimiter=', ', fmt='%.8g', 
           header='European Swaption Price Vector in Basis Points of the Notional')
np.savetxt(one_factor_HW_model.data_dir+'se_vector_RLNN.txt', 
           se_vector[0], delimiter=', ', fmt='%.8g', 
           header='European Swaption Price Standard Error Vector in Basis Points of the Notional')
np.savetxt(one_factor_HW_model.data_dir+'mean_abs_err_vector_RLNN.txt', 
           mean_abs_err_vector[0], delimiter=', ', fmt='%.8g', 
           header='European Swaption Price Absolute Error Vector')
np.savetxt(one_factor_HW_model.data_dir+'se_abs_err_vector_RLNN.txt', 
           se_abs_err_vector[0], delimiter=', ', fmt='%.8g', 
           header='European Swaption Price Standard Error of Absolute Error Vector')
np.savetxt(one_factor_HW_model.data_dir+'timing_vector_RLNN.txt', 
           timing_vector[0], delimiter=', ', fmt='%.8g', 
           header='European Swaption Price Function Timing Vector')
np.savetxt(one_factor_HW_model.data_dir+'MSE_vector_RLNN.txt', 
           MSE_vector, delimiter=', ', fmt='%.8g', 
           header='European Swaption Neural Network Fit MSE')

np.savetxt(one_factor_HW_model.data_dir+'mean_price_vector_MC.txt', 
           mean_price_vector[1], delimiter=', ', fmt='%.8g', 
           header='European Swaption Price Vector in Basis Points of the Notional')
np.savetxt(one_factor_HW_model.data_dir+'se_vector_MC.txt', 
           se_vector[1], delimiter=', ', fmt='%.8g', 
           header='European Swaption Price Standard Error Vector in Basis Points of the Notional')
np.savetxt(one_factor_HW_model.data_dir+'mean_abs_err_vector_MC.txt', 
           mean_abs_err_vector[1], delimiter=', ', fmt='%.8g', 
           header='European Swaption Price Absolute Error Vector')
np.savetxt(one_factor_HW_model.data_dir+'se_abs_err_vector_MC.txt', 
           se_abs_err_vector[1], delimiter=', ', fmt='%.8g', 
           header='European Swaption Price Standard Error of Absolute Error Vector')
np.savetxt(one_factor_HW_model.data_dir+'timing_vector_MC.txt', 
           timing_vector[1], delimiter=', ', fmt='%.8g', 
           header='European Swaption Price Function Timing Vector')

with open(one_factor_HW_model.data_dir + 'EuropeanRLNN_price_and_abs_err_results_final.txt', 
          'w') as f:
    f.write(f'Bachelier price: {MCEuropeanSwaption.exact_price_Bachelier}')
    f.write(f'\nmean_price_vector:\n{mean_price_vector}\n')
    f.write(f'\nse_vector:\n{se_vector}\n')
    f.write(f'\nmean_abs_err_vector:\n{mean_abs_err_vector}\n')
    f.write(f'\nse_abs_err_vector:\n{se_abs_err_vector}\n')
    f.write(f'\nMSE_vector:\n{MSE_vector}\n')
    
# Mean of means
with open(one_factor_HW_model.data_dir + 'EuropeanRLNN_mean_of_means_results.txt', 
          'w') as f:
    f.write('### RLNN ###')
    f.write(f'\nmean of mean prices:\n{np.mean(mean_price_vector[0], axis=1)}\n')
    f.write(f'\nSEM of mean prices:\n{st.sem(mean_price_vector[0], axis=1)}\n')
    f.write(f'\nmean of mean abs. errors:\n{np.mean(mean_abs_err_vector[0], axis=1)}\n')
    f.write(f'\nSEM of mean abs. errors:\n{st.sem(mean_abs_err_vector[0], axis=1)}\n')
    f.write(f'\nmean of MSEs:\n{np.mean(MSE_vector, axis=1)}\n')
    f.write(f'\nSEM of MSEs:\n{st.sem(MSE_vector, axis=1)}\n')
    f.write(f'\nmean of mean timings:\n{np.mean(timing_vector[0], axis=1)}\n')
    f.write(f'\nSEM of mean timings:\n{st.sem(timing_vector[0], axis=1)}\n\n')
    f.write('### Monte Carlo ###')
    f.write(f'\nmean of mean prices:\n{np.mean(mean_price_vector[1], axis=1)}\n')
    f.write(f'\nSEM of mean prices:\n{st.sem(mean_price_vector[1], axis=1)}\n')
    f.write(f'\nmean of mean abs. errors:\n{np.mean(mean_abs_err_vector[1], axis=1)}\n')
    f.write(f'\nSEM of mean abs. errors:\n{st.sem(mean_abs_err_vector[1], axis=1)}\n')
    f.write(f'\nmean of mean timings:\n{np.mean(timing_vector[1], axis=1)}\n')
    f.write(f'\nSEM of mean timings:\n{st.sem(timing_vector[1], axis=1)}\n\n')

#%% 
## SHUTDOWN COMMAND!! ##
# os.system("shutdown /s /t 1")