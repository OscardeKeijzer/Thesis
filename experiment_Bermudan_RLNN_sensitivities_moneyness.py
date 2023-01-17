# -*- coding: utf-8 -*-
# Imports
import copy
from datetime import datetime
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
from interest_rate_functions import (eval_annuity_terms, 
                                     eval_swap_rate,
                                     interpolate_zero_rate)
from least_squares_Monte_Carlo import (price_Bermudan_stock_option_LSM, 
                                       price_Bermudan_swaption_LSM_Q)
from neural_networks import ShallowFeedForwardNeuralNetwork
from one_factor_Hull_White_model import (eval_discount_factors,
                                         gen_one_factor_Hull_White_paths)
from plotting_functions import (plot_time_series, 
                                plot_one_factor_Hull_White_histogram)
from swaps import (price_forward_start_swap)
from swaptions import (price_European_swaption_exact_Jamshidian, 
                       price_European_swaption_MC_Q)
import tensorflow.keras as keras


#%% Initialize 1F-HW model instance:
tenor_structures = [np.arange(1., 6.+1), np.arange(5., 10.+1)]
moneynesses = [0.5, 2.]
n_moneynesses = len(moneynesses)
n_sims = 10

for tenor_structure in tenor_structures:
    tenor_notation = f'{int(tenor_structure[0])}Yx{int(tenor_structure[-1]-tenor_structure[0])}Y'
    exp_dir_name = f'Bermudan Swaption RLNN Sensitivities {tenor_notation} Moneyness'
    n_annual_trading_days = 253
    simulation_time = 30.
    time_0_rate = .001
    one_factor_HW_model = OneFactorHullWhiteModel(exp_dir_name, 
                                                  n_annual_trading_days, 
                                                  simulation_time, time_0_rate)
    one_factor_HW_model.construct_curves(True, 'flat', 'continuously-compounded', 
                                         0., False)
    
    # Short rate simulation parameters
    a_param = .01
    antithetic = True
    n_paths = 12000
    seed = None
    sigma = .01
    
    # Swaption parameters
    fixed_rate = .0305
    notional = 1.
    payoff_var = 'swap'
    plot = False
    swap_type = swaption_type = 'receiver'
    # tenor_structure = np.arange(5., 15.+1)
    units_basis_points = False
    r_simulation_time = tenor_structure[-1]
    
    # RLNN parameters
    n_epochs = 4500
    n_hidden_units = 64
    learning_rate = .0003
    # n_runs = 1
    train_frac = 2000/n_paths
    # price_vector = np.zeros(n_runs)
    # se_vector = np.zeros_like(price_vector)
    # abs_err_vector = np.zeros_like(price_vector)
    save_weights = False
    seed_biases = None
    seed_weights = None
    n_pricing_paths = int((1 - train_frac)*n_paths)
    
    # Determine the number of sensitivity evaluation times for the current Bermudan swaption
    eval_times = np.arange(0., tenor_structure[-1]-1., 1/12)
    n_eval_times = len(eval_times[~np.in1d(eval_times, tenor_structure)])
    
    # Determine the number of risk tenors relevant to the current Bermudan swaption
    max_idx = np.searchsorted(np.array(list(one_factor_HW_model.ISDA_SIMM_tenors_dict.values())), tenor_structure[-1])
    n_risk_tenors = len(np.array(list(one_factor_HW_model.ISDA_SIMM_tenors_dict.values()))[:max_idx+1])
    
    # Initialize array for storing results
    dVdR_RLNN_array = np.zeros((n_moneynesses, n_sims, n_eval_times, n_risk_tenors, n_pricing_paths))
    
    one_factor_HW_exp_dir = one_factor_HW_model.experiment_dir
    
    # Iterate over simulations to compute mean sensitivity profile
    for sim_count in range(n_sims):
        print(f'run {sim_count+1}\n')
        
        # Simulate short rates
        one_factor_HW_model.sim_short_rate_paths(a_param, antithetic, 
                                                 one_factor_HW_model.init_forward_c, 
                                                 one_factor_HW_model.init_forward_c_dfdT, 
                                                 n_paths, seed,'zero-mean', 
                                                 r_simulation_time, sigma, 'Euler', 
                                                 time_0_rate)
        
        
        # Iterate over moneynesses
        for moneyness in moneynesses:
            one_factor_HW_model.experiment_dir = one_factor_HW_exp_dir + f' {moneyness*100}%'
            one_factor_HW_model.data_dir = os.path.join(one_factor_HW_model.experiment_dir, 'Data\\')
            if not os.path.isdir(one_factor_HW_model.data_dir):
                os.makedirs(one_factor_HW_model.data_dir)
            one_factor_HW_model.figures_dir = os.path.join(one_factor_HW_model.experiment_dir, 'Figures\\')
            if not os.path.isdir(one_factor_HW_model.figures_dir):
                os.makedirs(one_factor_HW_model.figures_dir)
            
            # Replicate swaption
            BermudanRLNN = BermudanSwaptionRLNN(fixed_rate, one_factor_HW_model, notional, 
                                                moneyness, swaption_type, 
                                                tenor_structure, units_basis_points=units_basis_points)
            BermudanRLNN.replicate(None, 0., learn_rate=learning_rate, n_epochs=n_epochs, 
                                   n_hidden_nodes=n_hidden_units, seed_biases=seed_biases, 
                                   seed_weights=seed_weights, save_weights=save_weights, 
                                   test_fit=True, train_size=train_frac)
            BermudanRLNN.price_direct_estimator(0.)
            
            # Compute sensitivities
            starting_time = datetime.now()
            BermudanRLNN.eval_forward_sensitivities(eval_times, 'ISDA-SIMM')
            finish_time = datetime.now()
            
            # Store sensitivities for computing means
            for m_count, m in enumerate(moneynesses):
                if moneyness == m:
                    moneyness_idx = m_count
                
            dVdR_RLNN_array[moneyness_idx,sim_count] = BermudanRLNN.dVdR
            
            if sim_count > 0:
                plotRLNN =  BermudanSwaptionRLNN(fixed_rate, one_factor_HW_model, notional, 
                                                    moneyness, swaption_type, 
                                                    tenor_structure, units_basis_points=units_basis_points)
                plotRLNN.__dict__ = BermudanRLNN.__dict__.copy()
                # plotRLNN.dVdR = (np.concatenate([dVdR_RLNN_array[moneyness_idx,c] 
                #                                  for c in range(np.shape(dVdR_RLNN_array[:sim_count+1])[0])], axis=-1))
                plotRLNN.dVdR = np.concatenate(dVdR_RLNN_array[moneyness_idx,:sim_count+1], axis=-1)
                plotRLNN.plot_forward_sensitivities('RLNN', True, None, None, True)
                plotRLNN.write_sensitivities(True, True)
                # print(np.mean(dVdR_RLNN_array[moneyness_idx,sim_count], axis=(0,1)) - np.mean(plotRLNN.dVdR[:,:,int(100*sim_count):], axis=(0,1)))
                
            ## Remove this section later, only for obtaining missing MSEs in Table 5.8 
            ## of thesis report
            with open(one_factor_HW_model.data_dir + 'MSEs.txt', 
                      'a') as f:
                f.write(f'{BermudanRLNN.MSE[0]}, ')
            
            with open(one_factor_HW_model.data_dir + 'prices.txt', 
                      'a') as f:
                f.write(f'{BermudanRLNN.mean_direct_price_estimator}, ')
                
            with open(one_factor_HW_model.data_dir + 'runtimes.txt', 
                      'a') as f:
                f.write(f'{(finish_time - starting_time).total_seconds()}, ')
#%%
# print(np.shape(dVdR_RLNN_array))
# print(np.mean(dVdR_RLNN_array[1,-1], axis=(0,1)))
# print(np.shape(plotRLNN.dVdR))
# print(np.mean(plotRLNN.dVdR[:,:,90000:], axis=(0,1)))
                #%%
# time_0_swap_rate = eval_swap_rate(plotRLNN.model.init_zero_bond_c, 
#                                   plotRLNN.model.n_annual_trading_days, 
#                                   plotRLNN.swaption_type, 
#                                   plotRLNN.tenor_structure, 
#                                   0.)
# print(f'Moneyness: {BermudanRLNN.swap_rate_moneyness}')
# print(f'Time-zero swap rate: {time_0_swap_rate}')
# print(f'Fixed rate: {BermudanRLNN.fixed_rate}')
# print(f'Mean price: {BermudanRLNN.mean_direct_price_estimator}')
#%%
# plotRLNN =  BermudanSwaptionRLNN(fixed_rate, one_factor_HW_model, notional, 
#                                     swap_rate_moneyness, swaption_type, 
#                                     tenor_structure, units_basis_points=units_basis_points)
# plotRLNN.__dict__ = BermudanRLNN.__dict__.copy()
# plotRLNN.dVdR = np.concatenate([dVdR_RLNN_array[c] for c in range(np.shape(dVdR_RLNN_array[:sim_count+1])[0])], axis=-1)
# plotRLNN.plot_forward_sensitivities('RLNN', True, None, None, True)
#%%
# print(np.mean(plotRLNN.dVdR[:,7,:], axis=-1))
# print(np.array2string(plotRLNN.Delta_eval_times, separator=', '))
#%%
# plotRLNN.write_sensitivities(True, True)
#%% 
## SHUTDOWN COMMAND!! ##
# os.system("shutdown /s /t 1")