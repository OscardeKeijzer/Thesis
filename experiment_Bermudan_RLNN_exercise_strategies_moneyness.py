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
tenor_structures = [np.arange(1., 6.+1), np.arange(5., 15.+1)]
n_tenor_structures = len(tenor_structures)
# moneynesses = [0.5, 1., 2.]
# n_moneynesses = len(moneynesses)
n_sims = 5

for tenor_count, tenor_structure in enumerate(tenor_structures):
    tenor_notation = f'{int(tenor_structure[0])}Yx{int(tenor_structure[-1]-tenor_structure[0])}Y'
    exp_dir_name = f'Bermudan Swaption Exercise Strategies {tenor_notation} Moneyness'
    n_annual_trading_days = 253
    simulation_time = 30.
    time_0_rate = .001
    one_factor_HW_model = OneFactorHullWhiteModel(exp_dir_name, 
                                                  n_annual_trading_days, 
                                                  simulation_time, time_0_rate)
    one_factor_HW_model.construct_curves(False, 'flat', 'continuously-compounded', 
                                         0., False)
    
    if tenor_count == 0:
        moneynesses = [1.]
    elif tenor_count == 1:
        moneynesses = [0.5, 1., 2.]
    n_moneynesses = len(moneynesses)
    
    # Short rate simulation parameters
    a_param = .01
    antithetic = True
    n_paths = 12000
    seed = None
    sigma = .01
    
    # Swaption parameters
    fixed_rate = .0305
    notional = 1.
    plot = False
    swap_type = swaption_type = 'receiver'
    tenor = len(tenor_structure)
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
    
    # LSM parameters
    degree = 2
    payoff_var = 'swap'
    regression_series = 'power'
    
    
    # Initialize array for storing results
    live_paths_RLNN_array = np.zeros((n_moneynesses, n_sims, tenor-1, n_pricing_paths))
    live_paths_LSM_array = np.zeros_like(live_paths_RLNN_array)
    exercised_paths_RLNN_array = np.zeros_like(live_paths_RLNN_array)
    exercised_paths_LSM_array = np.zeros_like(live_paths_RLNN_array)
    
    one_factor_HW_exp_dir = one_factor_HW_model.experiment_dir
    
    # Iterate over simulations to determine exercise strategies
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
            
            # Replicate swaption and evaluate stopping times
            BermudanRLNN = BermudanSwaptionRLNN(fixed_rate, one_factor_HW_model, notional, 
                                                moneyness, swaption_type, 
                                                tenor_structure, units_basis_points=units_basis_points)
            BermudanRLNN.replicate(None, 0., learn_rate=learning_rate, n_epochs=n_epochs, 
                                   n_hidden_nodes=n_hidden_units, seed_biases=seed_biases, 
                                   seed_weights=seed_weights, save_weights=save_weights, 
                                   test_fit=False, train_size=train_frac)
            BermudanRLNN.price_direct_estimator(0.)
            BermudanRLNN.eval_exercise_strategies()
            
            # LSM price
            BermudanLSM = BermudanSwaptionLSM(one_factor_HW_model)
            BermudanLSM.price_Bermudan_swaption_LSM_Q(degree, fixed_rate, 
                                                      notional, payoff_var, 
                                                      False, False, False, 
                                                      regression_series, 
                                                      moneyness, swaption_type, 
                                                      tenor_structure, 0., 
                                                      units_basis_points, 
                                                      r_t_paths=BermudanRLNN.pricing_r_t_paths,
                                                      verbose=False, 
                                                      x_t_paths=BermudanRLNN.pricing_x_t_paths)
            
                
            # Store pathwise stopping times and exercised path indices
            for m_count, m in enumerate(moneynesses):
                if moneyness == m:
                    moneyness_idx = m_count
                
            live_paths_RLNN_array[moneyness_idx,sim_count] = BermudanRLNN.live_paths_array
            live_paths_LSM_array[moneyness_idx,sim_count] = BermudanLSM.live_paths_array
            exercised_paths_RLNN_array[moneyness_idx,sim_count] = BermudanRLNN.pathwise_stopping_times
            exercised_paths_LSM_array[moneyness_idx,sim_count] = BermudanLSM.pathwise_stopping_times
            
            if sim_count>0:
                # Histogram of exercise strategies
                
                fig, ax = plt.subplots(1, 1, figsize=(6,4))
                fig.suptitle(f'{tenor_notation} Bermudan {swaption_type.capitalize()} ' 
                             + f'Swaption {int(moneyness*100)}% Moneyness' 
                             + '\nMean Exercise Strategy for ' + f'{sim_count+1}' 
                             + r'$\times$' + f'{n_pricing_paths:,} Paths')
                bar_offset = .2
                mean_RLNN_live = np.sum(np.mean(live_paths_RLNN_array[moneyness_idx,:sim_count+1], axis=0), axis=1)
                SEM_RLNN_live = np.sum(st.sem(live_paths_RLNN_array[moneyness_idx,:sim_count+1], axis=0), axis=1)
                mean_RLNN_exercised = np.sum(np.mean(exercised_paths_RLNN_array[moneyness_idx,:sim_count+1], axis=0), axis=1)
                SEM_RLNN_exercised = np.sum(st.sem(exercised_paths_RLNN_array[moneyness_idx,:sim_count+1], axis=0), axis=1)
                mean_LSM_live = np.sum(np.mean(live_paths_LSM_array[moneyness_idx,:sim_count+1], axis=0), axis=1)
                SEM_LSM_live = np.sum(st.sem(live_paths_LSM_array[moneyness_idx,:sim_count+1], axis=0), axis=1)
                mean_LSM_exercised = np.sum(np.mean(exercised_paths_LSM_array[moneyness_idx,:sim_count+1], axis=0), axis=1)
                SEM_LSM_exercised = np.sum(st.sem(exercised_paths_LSM_array[moneyness_idx,:sim_count+1], axis=0), axis=1)
                
                ax.bar(tenor_structure[:-1]-bar_offset, mean_RLNN_live, 
                        # np.count_nonzero(RLNN_stopping_times, axis=-1), 
                        # np.sum(np.mean(live_paths_RLNN_array[0,:sim_count+1], axis=0), axis=1),
                        yerr=SEM_RLNN_live, width=.4, color='C0', alpha=1., 
                        label='RLNN\nLive Paths')
                ax.bar(tenor_structure[:-1]+bar_offset, mean_LSM_live,
                       # np.count_nonzero(LSM_stopping_times, axis=-1), 
                       # np.sum(np.mean(live_paths_LSM_array[0,:sim_count+1], axis=0), axis=1),
                       yerr=SEM_LSM_live, width=.4, color='C1', alpha=1., 
                       label='LSM\nLive Paths')
                ax.bar(tenor_structure[:-1]-bar_offset, mean_RLNN_exercised, 
                       # np.count_nonzero(RLNN_exercised_paths, axis=-1), 
                       # np.sum(np.mean(exercised_paths_RLNN_array[0,:sim_count+1], axis=0), axis=1),
                       yerr=SEM_RLNN_exercised, width=.4, color='black', 
                       alpha=.25, hatch='/', label='Exercised')
                ax.bar(tenor_structure[:-1]+bar_offset, mean_LSM_exercised,
                       # np.count_nonzero(LSM_exercised_paths, axis=-1), 
                       # np.sum(np.mean(exercised_paths_LSM_array[0,:sim_count+1], axis=0), axis=1),
                       yerr=SEM_LSM_exercised, width=.4, color='black', alpha=.25, hatch='/')
                # ax.hist(LSM_stopping_times, bins=tenor_structure[:-1], alpha=0.5, 
                #         color='C1', label='LSM')
                # counts, bins = np.histogram(np.count_nonzero(RLNN_stopping_times, axis=-1))
                # ax.stairs(counts, bins)
                ax.set_xlabel('Monitor Date (years)')
                ax.set_ylabel('Number of Paths')
                ax.set_xticks(tenor_structure[:-1])
                ax.set_xticklabels((tenor_structure[:-1].astype(int)).astype(str))
                plt.legend(loc='best')
                plt.tight_layout()
                
                plot_name = (f'{tenor_notation} Bermudan {swaption_type.capitalize()}' 
                             + f'Swaption {int(100*moneyness)} Moneyness Stopping Times Histogram')
                file_dir_and_name = str(one_factor_HW_model.figures_dir 
                                        + (plot_name.replace(' ','_'))  + '-' 
                                        + datetime.now().strftime('%Y-%m-%d_%H%M%S'))
                plt.savefig(file_dir_and_name, bbox_inches='tight')
                print('\nPlot was saved to ' + file_dir_and_name + '.png')
                plt.show()
                
                # Write results
                with open(one_factor_HW_model.data_dir + 'mean_live_paths_RLNN.txt', 
                          'a') as f:
                    f.write('Mean number of live paths for ' 
                            + f'{(sim_count+1)}*{n_pricing_paths} total paths ' 
                            + f'at exercise dates {tenor_structure[:-1]}:\n'
                            + f'{mean_RLNN_live}\n')
                with open(one_factor_HW_model.data_dir + 'mean_exercised_paths_RLNN.txt', 
                          'a') as f:
                    f.write('Mean number of exercised paths for ' 
                            + f'{(sim_count+1)}*{n_pricing_paths} total paths ' 
                            + f'at exercise dates {tenor_structure[:-1]}:\n'
                            + f'{mean_RLNN_exercised}\n')
                    
                with open(one_factor_HW_model.data_dir + 'mean_live_paths_LSM.txt', 
                          'a') as f:
                    f.write('Mean number of live paths for ' 
                            + f'{(sim_count+1)}*{n_pricing_paths} total paths ' 
                            + f'at exercise dates {tenor_structure[:-1]}:\n'
                            + f'{mean_LSM_live}\n')
                with open(one_factor_HW_model.data_dir + 'mean_exercised_paths_LSM.txt', 
                          'a') as f:
                    f.write('Mean number of exercised paths for ' 
                            + f'{(sim_count+1)}*{n_pricing_paths} total paths ' 
                            + f'at exercise dates {tenor_structure[:-1]}:\n'
                            + f'{mean_LSM_exercised}\n')
#%%
print(np.shape(live_paths_RLNN_array))
print(np.shape(live_paths_RLNN_array[0,:]))
print(np.shape(np.sum(live_paths_RLNN_array[0,:], axis=-1)))
print(np.sum(live_paths_RLNN_array[0,:], axis=-1))
print(np.mean(np.sum(live_paths_RLNN_array[0,:], axis=-1), axis=0))
# print(st.sem(np.sum(live_paths_RLNN_array[0,:], axis=-1), axis=1)[0])

#%%
for moneyness in moneynesses:
    one_factor_HW_model.experiment_dir = one_factor_HW_exp_dir + f' {moneyness*100}%'
    one_factor_HW_model.data_dir = os.path.join(one_factor_HW_model.experiment_dir, 'Data\\')
    if not os.path.isdir(one_factor_HW_model.data_dir):
        os.makedirs(one_factor_HW_model.data_dir)
    one_factor_HW_model.figures_dir = os.path.join(one_factor_HW_model.experiment_dir, 'Figures\\')
    if not os.path.isdir(one_factor_HW_model.figures_dir):
        os.makedirs(one_factor_HW_model.figures_dir)
    
    for m_count, m in enumerate(moneynesses):
        if moneyness == m:
            moneyness_idx = m_count
            
    print(moneyness, moneyness_idx)            
    # Histogram of exercise strategies
    # RLNN_stopping_times = np.concatenate(live_paths_RLNN_array[moneyness_idx,:sim_count+1], axis=-1)
    # LSM_stopping_times = np.concatenate(live_paths_LSM_array[moneyness_idx,:sim_count+1], axis=-1)
    # RLNN_exercised_paths = np.concatenate(exercised_paths_RLNN_array[moneyness_idx,:sim_count+1], axis=-1)
    # LSM_exercised_paths = np.concatenate(exercised_paths_LSM_array[moneyness_idx,:sim_count+1], axis=-1)
    # print(f'Shape of RLNN_stopping_times: {np.shape(RLNN_stopping_times)}')
    # print(f'Non-zero RLNN_stopping_times: {np.count_nonzero(RLNN_stopping_times, axis=-1)}')
    
#%%
fig, ax = plt.subplots(1, 1, figsize=(6,4))
fig.suptitle(f'{tenor_notation} Bermudan {swaption_type.capitalize()} ' 
             + f'Swaption {int(moneyness*100)}% Moneyness' 
             + '\nMean Exercise Strategy for ' + f'{sim_count+1}' 
             + r'$\times$' + f'{n_pricing_paths:,} Paths')
bar_offset = .2
''
mean_RLNN_live = np.mean(np.sum(live_paths_RLNN_array[0,:2], axis=-1), axis=0)
SEM_RLNN_live = st.sem(np.sum(live_paths_RLNN_array[0,:2], axis=-1), axis=0)
mean_RLNN_exercised = np.mean(np.sum(exercised_paths_RLNN_array[0,:2], axis=-1), axis=0)
SEM_RLNN_exercised = st.sem(np.sum(exercised_paths_RLNN_array[0,:2], axis=-1), axis=0)
mean_LSM_live = np.mean(np.sum(live_paths_LSM_array[0,:2], axis=-1), axis=0)
SEM_LSM_live = st.sem(np.sum(live_paths_RLNN_array[0,:2], axis=-1), axis=0)
mean_LSM_exercised = np.mean(np.sum(exercised_paths_LSM_array[0,:2], axis=-1), axis=0)
SEM_LSM_exercised = st.sem(np.sum(exercised_paths_LSM_array[0,:2], axis=-1), axis=0)

# print(mean_RLNN_live)
# assert 1==2

ax.bar(tenor_structure[:-1]-bar_offset, mean_RLNN_live, 
        # np.count_nonzero(RLNN_stopping_times, axis=-1), 
        # np.sum(np.mean(live_paths_RLNN_array[0,:sim_count+1], axis=0), axis=1),
        yerr=1.96*SEM_RLNN_live, width=.4, color='C0', alpha=1., 
        label='RLNN\nLive Paths')
ax.bar(tenor_structure[:-1]+bar_offset, mean_LSM_live,
       # np.count_nonzero(LSM_stopping_times, axis=-1), 
       # np.sum(np.mean(live_paths_LSM_array[0,:sim_count+1], axis=0), axis=1),
       yerr=1.96*SEM_LSM_live, width=.4, color='C1', alpha=1., 
       label='LSM\nLive Paths')
ax.bar(tenor_structure[:-1]-bar_offset, mean_RLNN_exercised, 
       # np.count_nonzero(RLNN_exercised_paths, axis=-1), 
       # np.sum(np.mean(exercised_paths_RLNN_array[0,:sim_count+1], axis=0), axis=1),
       yerr=1.96*SEM_RLNN_exercised, width=.4, color='black', 
       alpha=.25, hatch='/', label='Exercised')
ax.bar(tenor_structure[:-1]+bar_offset, mean_LSM_exercised,
       # np.count_nonzero(LSM_exercised_paths, axis=-1), 
       # np.sum(np.mean(exercised_paths_LSM_array[0,:sim_count+1], axis=0), axis=1),
       yerr=1.96*SEM_LSM_exercised, width=.4, color='black', alpha=.25, hatch='/')
# ax.hist(LSM_stopping_times, bins=tenor_structure[:-1], alpha=0.5, 
#         color='C1', label='LSM')
# counts, bins = np.histogram(np.count_nonzero(RLNN_stopping_times, axis=-1))
# ax.stairs(counts, bins)
ax.set_xlabel('Monitor Date (years)')
ax.set_ylabel('Number of Paths')
ax.set_xticks(tenor_structure[:-1])
ax.set_xticklabels((tenor_structure[:-1].astype(int)).astype(str))
plt.legend(loc='best')
plt.tight_layout()

plot_name = (f'{tenor_notation} Bermudan {swaption_type.capitalize()}' 
             + f'Swaption {int(100*moneyness)} Moneyness Stopping Times Histogram')
file_dir_and_name = str(one_factor_HW_model.figures_dir 
                        + (plot_name.replace(' ','_'))  + '-' 
                        + datetime.now().strftime('%Y-%m-%d_%H%M%S'))
plt.savefig(file_dir_and_name, bbox_inches='tight')
# print('\nPlot was saved to ' + file_dir_and_name + '.png')
plt.show()

# Write results
with open(one_factor_HW_model.data_dir + 'mean_live_paths_RLNN.txt', 
          'a') as f:
    f.write('Mean number of live paths for ' 
            + f'{(sim_count+1)}*{n_pricing_paths} total paths ' 
            + f'at exercise dates {tenor_structure[:-1]}:\n'
            + f'{mean_RLNN_live}\n')
with open(one_factor_HW_model.data_dir + 'mean_exercised_paths_RLNN.txt', 
          'a') as f:
    f.write('Mean number of exercised paths for ' 
            + f'{(sim_count+1)}*{n_pricing_paths} total paths ' 
            + f'at exercise dates {tenor_structure[:-1]}:\n'
            + f'{mean_RLNN_exercised}\n')
    
with open(one_factor_HW_model.data_dir + 'mean_live_paths_LSM.txt', 
          'a') as f:
    f.write('Mean number of live paths for ' 
            + f'{(sim_count+1)}*{n_pricing_paths} total paths ' 
            + f'at exercise dates {tenor_structure[:-1]}:\n'
            + f'{mean_LSM_live}\n')
with open(one_factor_HW_model.data_dir + 'mean_exercised_paths_LSM.txt', 
          'a') as f:
    f.write('Mean number of exercised paths for ' 
            + f'{(sim_count+1)}*{n_pricing_paths} total paths ' 
            + f'at exercise dates {tenor_structure[:-1]}:\n'
            + f'{mean_LSM_exercised}\n')

#%% 
## SHUTDOWN COMMAND!! ##
#%% 
## SHUTDOWN COMMAND!! ##
os.system("shutdown /h /t 1")