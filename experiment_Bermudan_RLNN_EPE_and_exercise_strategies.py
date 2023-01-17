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
swap_type = swaption_type = 'receiver'
tenor_structures = [np.arange(1., 4.+1)] #[np.arange(1., 6.+1), np.arange(5., 15.+1)]
moneynesses = [1.] # [0.5, 1., 2.]
n_moneynesses = len(moneynesses)
n_sims = 2

for tenor_count, tenor_structure in enumerate(tenor_structures):
    tenor_notation = f'{int(tenor_structure[0])}Yx{int(tenor_structure[-1]-tenor_structure[0])}Y'
    exp_dir_name = f'RLNN {tenor_notation} Bermudan {swaption_type.capitalize()} Swaption EPE Profiles and Exercise FINAL TEST'
    n_annual_trading_days = 253
    simulation_time = 30.
    time_0_rate = .01
    one_factor_HW_model = OneFactorHullWhiteModel(exp_dir_name, 
                                                  n_annual_trading_days, 
                                                  simulation_time, time_0_rate)
    one_factor_HW_model.construct_curves(False, 'flat', 'continuously-compounded', 
                                         0., False)
    
    # # # Specify moneyness per tenor structure
    # if tenor_count == 0:
    #     moneynesses = [1.]
    # elif tenor_count == 1:
    #     moneynesses = [0.5, 1., 2.]
    # n_moneynesses = len(moneynesses)
    
    # Short rate simulation parameters
    a_param = .012
    antithetic = True
    n_paths = 12000
    seed = None
    sigma = .020
    
    # Swaption parameters
    fixed_rate = .0305
    notional = 1.
    payoff_var = 'swap'
    plot = False
    tenor = len(tenor_structure)
    units_basis_points = False
    r_simulation_time = tenor_structure[-1]
    
    # RLNN parameters
    n_epochs = 4500
    n_hidden_units = 64
    learning_rate = .0003
    train_frac = 2000/n_paths
    save_weights = False
    seed_biases = None
    seed_weights = None
    n_pricing_paths = int((1 - train_frac)*n_paths)
    
    # LSM parameters
    degree = 2
    payoff_var = 'swap'
    regression_series = 'power'
    
    # Determine the number of EPE profile evaluation times for the current Bermudan swaption
    eval_times = np.arange(0., tenor_structure[-2], 1/12)
    n_eval_times = len(eval_times)
    
    # Initialize array for storing EPE and exercise strategy results
    EPE_RLNN_array = np.zeros((n_moneynesses, n_sims, n_eval_times, n_pricing_paths))
    EPE_LSM_array = np.zeros_like(EPE_RLNN_array)
    live_paths_RLNN_array = np.zeros((n_moneynesses, n_sims, tenor-1, n_pricing_paths))
    live_paths_LSM_array = np.zeros_like(live_paths_RLNN_array)
    exercised_paths_RLNN_array = np.zeros_like(live_paths_RLNN_array)
    exercised_paths_LSM_array = np.zeros_like(live_paths_RLNN_array)
    
    one_factor_HW_exp_dir = one_factor_HW_model.experiment_dir
    
    # Iterate over simulations to compute mean EPE profile
    for sim_count in range(n_sims):
        print(f'run {sim_count+1}\n')
        
        # Simulate short rates
        one_factor_HW_model.sim_short_rate_paths(a_param, n_paths, 
                                                 r_simulation_time, sigma)
        
        # Iterate over moneynesses
        for moneyness in moneynesses:
            one_factor_HW_model.experiment_dir = os.path.join(one_factor_HW_exp_dir, f'{int(moneyness*100)}% Moneyness')
            if not os.path.isdir(one_factor_HW_model.experiment_dir):
                os.makedirs(one_factor_HW_model.experiment_dir)
            one_factor_HW_model.data_dir = os.path.join(one_factor_HW_model.experiment_dir, 'Data\\')
            if not os.path.isdir(one_factor_HW_model.data_dir):
                os.makedirs(one_factor_HW_model.data_dir)
            one_factor_HW_model.figures_dir = os.path.join(one_factor_HW_model.experiment_dir, 'Figures\\')
            if not os.path.isdir(one_factor_HW_model.figures_dir):
                os.makedirs(one_factor_HW_model.figures_dir)
            
            # Replicate swaption using RLNN
            BermudanRLNN = BermudanSwaptionRLNN(fixed_rate, one_factor_HW_model, 
                                                notional, moneyness, 
                                                swaption_type, tenor_structure)
            BermudanRLNN.replicate(None, 0., learn_rate=learning_rate, n_epochs=n_epochs, 
                                   n_hidden_nodes=n_hidden_units, seed_biases=seed_biases, 
                                   seed_weights=seed_weights, save_weights=save_weights, 
                                   test_fit=False, train_size=train_frac)
            BermudanRLNN.price_direct_estimator(0.)
            
            # Price swaption using LSM in order to obtain LSM exercise strategies
            BermudanLSM = BermudanSwaptionLSM(one_factor_HW_model)
            BermudanLSM.price_Bermudan_swaption_LSM_Q(degree, fixed_rate, moneyness, 
                                                      notional, swaption_type, 
                                                      tenor_structure, 0., 
                                                      units_basis_points, 
                                                      r_t_paths=BermudanRLNN.pricing_r_t_paths,
                                                      verbose=False, 
                                                      x_t_paths=BermudanRLNN.pricing_x_t_paths)
            
            # Write prices and MSEs
            with open(one_factor_HW_model.data_dir + 'MSEs.txt', 
                      'a') as f:
                f.write(f'{BermudanRLNN.MSE[0]}, ')
            
            with open(one_factor_HW_model.data_dir + 'prices.txt', 
                      'a') as f:
                f.write(f'{BermudanRLNN.mean_direct_price_estimator}, ')
                
            # Compute EPE profiles
            starting_time = datetime.now()
            BermudanRLNN.eval_EPE_profile(True, eval_times)
            finish_time = datetime.now()
            
            with open(one_factor_HW_model.data_dir + 'runtimes.txt', 
                      'a') as f:
                f.write(f'{(finish_time - starting_time).total_seconds()}, ')
            
            # Store EPE and exercise strategy results
            for m_count, m in enumerate(moneynesses):
                if moneyness == m:
                    moneyness_idx = m_count
                
            EPE_RLNN_array[moneyness_idx,sim_count] = BermudanRLNN.EPE_RLNN
            EPE_LSM_array[moneyness_idx,sim_count] = BermudanRLNN.EPE_LSM
            live_paths_RLNN_array[moneyness_idx,sim_count] = BermudanRLNN.live_paths_array
            live_paths_LSM_array[moneyness_idx,sim_count] = BermudanLSM.live_paths_array
            exercised_paths_RLNN_array[moneyness_idx,sim_count] = BermudanRLNN.pathwise_stopping_times
            exercised_paths_LSM_array[moneyness_idx,sim_count] = BermudanLSM.pathwise_stopping_times
            
            # Plot and write intermediate and final results
            if sim_count >= 0:
                plotRLNN =  BermudanSwaptionRLNN(fixed_rate, one_factor_HW_model, notional, 
                                                    moneyness, swaption_type, 
                                                    tenor_structure, units_basis_points=units_basis_points)
                plotRLNN.__dict__ = BermudanRLNN.__dict__.copy()
                plotRLNN.EPE_RLNN = np.concatenate(EPE_RLNN_array[moneyness_idx,:sim_count+1], axis=-1)
                plotRLNN.EPE_LSM = np.concatenate(EPE_LSM_array[moneyness_idx,:sim_count+1], axis=-1)
                plotRLNN.plot_EPE_profile('compare', True, units_basis_points=True)
                plotRLNN.write_EPE_profile(True, 'both', True)
                
                # Plot and write exercise strategies
                fig, ax = plt.subplots(1, 1, figsize=(6,4))
                fig.suptitle(f'{tenor_notation} Bermudan {swaption_type.capitalize()} ' 
                             + f'Swaption {int(moneyness*100)}% Moneyness' 
                             + '\nMean Exercise Strategy for ' + f'{sim_count+1}' 
                             + r'$\times$' + f'{n_pricing_paths:,} Paths')
                bar_offset = .2
                mean_RLNN_live = np.mean(np.sum(live_paths_RLNN_array[moneyness_idx,:sim_count+1], axis=-1), axis=0)
                SEM_RLNN_live = st.sem(np.sum(live_paths_RLNN_array[moneyness_idx,:sim_count+1], axis=-1), axis=0)
                mean_RLNN_exercised = np.mean(np.sum(exercised_paths_RLNN_array[moneyness_idx,:sim_count+1], axis=-1), axis=0)
                SEM_RLNN_exercised = st.sem(np.sum(exercised_paths_RLNN_array[moneyness_idx,:sim_count+1], axis=-1), axis=0)
                mean_LSM_live = np.mean(np.sum(live_paths_LSM_array[moneyness_idx,:sim_count+1], axis=-1), axis=0)
                SEM_LSM_live = st.sem(np.sum(live_paths_RLNN_array[moneyness_idx,:sim_count+1], axis=-1), axis=0)
                mean_LSM_exercised = np.mean(np.sum(exercised_paths_LSM_array[moneyness_idx,:sim_count+1], axis=-1), axis=0)
                SEM_LSM_exercised = st.sem(np.sum(exercised_paths_LSM_array[moneyness_idx,:sim_count+1], axis=-1), axis=0)
                
                ax.bar(tenor_structure[:-1]-bar_offset, mean_RLNN_live, 
                        yerr=1.96*SEM_RLNN_live, width=.4, color='C0', alpha=1., 
                        label='RLNN\nLive Paths')
                ax.bar(tenor_structure[:-1]+bar_offset, mean_LSM_live,
                       yerr=1.96*SEM_LSM_live, width=.4, color='C1', alpha=1., 
                       label='LSM\nLive Paths')
                ax.bar(tenor_structure[:-1]-bar_offset, mean_RLNN_exercised, 
                       yerr=1.96*SEM_RLNN_exercised, width=.4, color='black', 
                       alpha=.25, hatch='/', label='Exercised')
                ax.bar(tenor_structure[:-1]+bar_offset, mean_LSM_exercised,
                       yerr=1.96*SEM_LSM_exercised, width=.4, color='black', alpha=.25, hatch='/')
                ax.set_xlabel('Monitor Date (years)')
                ax.set_xticks(tenor_structure[:-1])
                ax.set_xticklabels((tenor_structure[:-1].astype(int)).astype(str))
                ax.set_ylabel('Number of Paths')
                ax.set_yticklabels(['{:,}'.format(int(y)) for y in ax.get_yticks().tolist()])
                plt.legend(loc='best')
                plt.tight_layout()
                
                plot_name = (f'{tenor_notation} Bermudan {swaption_type.capitalize()}' 
                             + f'Swaption {int(100*moneyness)} Moneyness Stopping Times Histogram')
                file_dir_and_name = str(one_factor_HW_model.figures_dir 
                                        + (plot_name.replace(' ','_'))  + '-' 
                                        + datetime.now().strftime('%Y-%m-%d_%H%M%S'))
                plt.savefig(file_dir_and_name, bbox_inches='tight')
                plt.show()

                # Write results
                with open(one_factor_HW_model.data_dir + 'mean_SEM_live_paths_RLNN.txt', 
                          'a') as f:
                    f.write('Mean and SEM of number of live paths for ' 
                            + f'{(sim_count+1)}*{n_pricing_paths} total paths ' 
                            + f'at exercise dates {tenor_structure[:-1]}:\n'
                            + f'{mean_RLNN_live}\n{SEM_RLNN_live}\n\n')
                with open(one_factor_HW_model.data_dir + 'mean_SEM_exercised_paths_RLNN.txt', 
                          'a') as f:
                    f.write('Mean and SEM of number of exercised paths for ' 
                            + f'{(sim_count+1)}*{n_pricing_paths} total paths ' 
                            + f'at exercise dates {tenor_structure[:-1]}:\n'
                            + f'{mean_RLNN_exercised}\n{SEM_RLNN_exercised}\n\n')
                    
                with open(one_factor_HW_model.data_dir + 'mean_SEM_live_paths_LSM.txt', 
                          'a') as f:
                    f.write('Mean and SEM of number of live paths for ' 
                            + f'{(sim_count+1)}*{n_pricing_paths} total paths ' 
                            + f'at exercise dates {tenor_structure[:-1]}:\n'
                            + f'{mean_LSM_live}\n{SEM_LSM_live}\n\n')
                with open(one_factor_HW_model.data_dir + 'mean_SEM_exercised_paths_LSM.txt', 
                          'a') as f:
                    f.write('Mean and SEM of number of exercised paths for ' 
                            + f'{(sim_count+1)}*{n_pricing_paths} total paths ' 
                            + f'at exercise dates {tenor_structure[:-1]}:\n'
                            + f'{mean_LSM_exercised}\n{SEM_LSM_exercised}\n\n')
                    
#%% 
## SHUTDOWN COMMAND!! ##
# os.system("shutdown /s /t 1")