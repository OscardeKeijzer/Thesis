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
tenor_structures = [np.arange(1., 6.+1), np.arange(5., 10.+1), np.arange(5., 15.+1)]

for tenor_structure in tenor_structures:
    tenor_notation = f'{int(tenor_structure[0])}Yx{int(tenor_structure[-1]-tenor_structure[0])}Y'
    exp_dir_name = f'European Swaption RLNN Sensitivities {tenor_notation}'
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
    n_paths = 11500
    r_simulation_time = 6.
    seed = None
    sigma = .01
    n_sims = 10
    
    # Swaption parameters
    fixed_rate = .0305
    notional = 1.
    payoff_var = 'swap'
    plot = False
    swap_rate_moneyness = 1.
    swap_type = swaption_type = 'receiver'
    # tenor_structure = np.arange(5., 15.+1)
    units_basis_points = False
    r_simulation_time = tenor_structure[-1]
    
    # RLNN parameters
    n_epochs = 2000
    n_hidden_units = 32
    learning_rate = .0004
    # n_runs = 1
    train_frac = 1500/n_paths
    # price_vector = np.zeros(n_runs)
    # se_vector = np.zeros_like(price_vector)
    # abs_err_vector = np.zeros_like(price_vector)
    save_weights = True
    seed_biases = None
    seed_weights = None
    n_pricing_paths = int((1 - train_frac)*n_paths)
    
    # Determine the number of sensitivity evaluation times for the current swaption
    eval_times = np.arange(0., tenor_structure[0], 1/12)
    n_eval_times = len(eval_times[~np.in1d(eval_times, tenor_structure)])
    
    # Determine the number of risk tenors relevant to the current swaption
    max_idx = np.searchsorted(np.array(list(one_factor_HW_model.ISDA_SIMM_tenors_dict.values())), tenor_structure[-1])
    n_risk_tenors = len(np.array(list(one_factor_HW_model.ISDA_SIMM_tenors_dict.values()))[:max_idx+1])
    
    # Initialize arrays for storing results
    dVdR_RLNN_array = np.zeros((n_sims, n_eval_times, n_risk_tenors, n_pricing_paths))
    dVdR_Bachelier_array = np.zeros_like(dVdR_RLNN_array)
    
    # Iterate over simulations to compute sensitivity profile
    for sim_count in range(n_sims):
        print(f'run {sim_count+1}\n')
        
        # Simulate short rates
        one_factor_HW_model.sim_short_rate_paths(a_param, antithetic, 
                                                 one_factor_HW_model.init_forward_c, 
                                                 one_factor_HW_model.init_forward_c_dfdT, 
                                                 n_paths, seed,'zero-mean', 
                                                 r_simulation_time, sigma, 'Euler', 
                                                 time_0_rate)
        
        # Replicate swaption
        EuropeanRLNN = EuropeanSwaptionRLNN(fixed_rate, one_factor_HW_model, notional, 
                                            swap_rate_moneyness, swaption_type, 
                                            tenor_structure, units_basis_points=units_basis_points)
        EuropeanRLNN.replicate(None, 0., learn_rate=learning_rate, n_epochs=n_epochs, 
                               n_hidden_nodes=n_hidden_units, seed_biases=seed_biases, 
                               seed_weights=seed_weights, save_weights=save_weights, 
                               test_fit=True, train_size=train_frac)
        EuropeanRLNN.price_direct_estimator(0.)
        
        # Compute sensitivities
        starting_time = datetime.now()
        EuropeanRLNN.eval_forward_sensitivities(eval_times, 'ISDA-SIMM')
        finish_time = datetime.now()
        
        # Store sensitivities for computing means
        dVdR_RLNN_array[sim_count] = EuropeanRLNN.dVdR_RLNN
        dVdR_Bachelier_array[sim_count] = EuropeanRLNN.dVdR_Bachelier
        
        if sim_count > 0:
            # dVdR_RLNN_array = np.mean(dVdR_RLNN_array[:sim_count+1], axis=(0))
            plotRLNN =  EuropeanSwaptionRLNN(fixed_rate, one_factor_HW_model, notional, 
                                                swap_rate_moneyness, swaption_type, 
                                                tenor_structure, units_basis_points=units_basis_points)
            plotRLNN.__dict__ = EuropeanRLNN.__dict__.copy()
            plotRLNN.dVdR_RLNN = np.concatenate([dVdR_RLNN_array[c] for c in range(np.shape(dVdR_RLNN_array[:sim_count+1])[0])], axis=-1)
            plotRLNN.dVdR_Bachelier = np.concatenate([dVdR_Bachelier_array[c] for c in range(np.shape(dVdR_Bachelier_array[:sim_count+1])[0])], axis=-1)
            plotRLNN.plot_forward_sensitivities('RLNN', True, True, None, True)
            plotRLNN.plot_forward_sensitivities('compare', True, True, None, True)
            plotRLNN.write_sensitivities('RLNN', True, True)
            plotRLNN.write_sensitivities('Bachelier', True, True)
            
        ## Remove this section later, only for obtaining missing MSEs in Table 5.8 
        ## of thesis report
        with open(one_factor_HW_model.data_dir + 'MSEs.txt', 
                  'a') as f:
            f.write(f'{EuropeanRLNN.MSE}, ')
        
        with open(one_factor_HW_model.data_dir + 'prices.txt', 
                  'a') as f:
            f.write(f'{EuropeanRLNN.mean_direct_price_estimator}, ')
            
        with open(one_factor_HW_model.data_dir + 'runtimes.txt', 
                  'a') as f:
            f.write(f'{(finish_time - starting_time).total_seconds()}, ')
