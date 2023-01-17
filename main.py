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
from interest_rate_functions import (eval_cont_comp_spot_rate, 
                                     eval_annuity_terms, 
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
n_annual_trading_days = 253
simulation_time = 30.
time_0_rate = .001
one_factor_HW_model = OneFactorHullWhiteModel(None, n_annual_trading_days, 
                                              simulation_time, time_0_rate)
one_factor_HW_model.construct_curves(True, 'flat', 'continuously-compounded', 
                                     0., False)

#%%
print(len(one_factor_HW_model.init_zero_bond_c))

#%% Simulate short rates
a_param = .01
antithetic = True
n_paths = 12000
r_simulation_time = 30.
seed = None
sigma = .01
one_factor_HW_model.sim_short_rate_paths(a_param, n_paths, r_simulation_time, 
                                         sigma, antithetic=True, verbose=True)

#%%
print(np.shape(one_factor_HW_model.x_t_paths))

#%% Bump yield curve
one_factor_HW_model.bump_yield_curve('ISDA-SIMM 5Y', plot_curve=True)
#%% Check bumped yield curve
time_5_idx = int(5*one_factor_HW_model.n_annual_trading_days)
time_10_idx = int(10*one_factor_HW_model.n_annual_trading_days)
time_15_idx = int(15*one_factor_HW_model.n_annual_trading_days)
print(one_factor_HW_model.init_zero_coupon_c_bumped[time_5_idx])
print(one_factor_HW_model.init_zero_coupon_c_bumped[time_5_idx+1])
print(one_factor_HW_model.init_zero_coupon_c_bumped[time_10_idx])
print(one_factor_HW_model.init_zero_coupon_c_bumped[time_15_idx-1])
print(one_factor_HW_model.init_zero_coupon_c_bumped[time_15_idx])
#%% Read previously simulated short rates
one_factor_HW_model.r_t_paths = read_Parquet_data(one_factor_HW_model.input_dir
                                             +'short_rates_f(0,t)=.001_N=200000_T=20\\1F-HW_short_rate_paths_array_zero-mean_Euler_antithetic-2022-08-25_140653.parquet')
one_factor_HW_model.x_t_paths = read_Parquet_data(one_factor_HW_model.input_dir
                                             +'short_rates_f(0,t)=.001_N=200000_T=20\\1F-HW_zero_mean_process_paths_array_zero-mean_Euler_antithetic-2022-08-25_140653.parquet')
one_factor_HW_model.n_paths = np.shape(one_factor_HW_model.x_t_paths)[1]
#%% Plot short rates
one_factor_HW_model.plot_short_rate_paths(r_simulation_time, 'both')
#%% Log short rates
one_factor_HW_model.log_short_rate_paths()
#%% Zero-coupon bond curve
ZCB_curve = construct_zero_coupon_bonds_curve(a_param, one_factor_HW_model.experiment_dir, 
                                  30., n_annual_trading_days, True, 
                                  one_factor_HW_model.r_t_paths, sigma, 0., 
                                  one_factor_HW_model.time_0_f_curve, 
                                  one_factor_HW_model.time_0_P_curve, 
                                  one_factor_HW_model.x_t_paths)
#%%
# print(np.shape(one_factor_HW_model.time_0_P_curve), 
#       np.shape(ZCB_curve))
for count in range(31):
    idx = int(count*253)
    print(one_factor_HW_model.time_0_P_curve[idx])
    print(np.mean(ZCB_curve[idx]))
    print('\n')
#%% Price zero-bond options
maturity_bond = 6.
maturity_option = 5.
option_type = 'call'
strike_price = .1
time_t = 0.
t_idx = int(time_t*n_annual_trading_days)
T_idx = int(maturity_option*n_annual_trading_days)
units_basis_points = False

exact_price = price_zero_coupon_bond_option_exact(a_param, maturity_bond, 
                                                  maturity_option, 
                                                  n_annual_trading_days, 
                                                  option_type, 
                                                  one_factor_HW_model.r_t_paths, 
                                                  sigma, strike_price, time_t, 
                                                  one_factor_HW_model.time_0_f_curve, 
                                                  one_factor_HW_model.time_0_P_curve, 
                                                  units_basis_points, 
                                                  None, 
                                                  one_factor_HW_model.x_t_paths)
MC_price_func = price_zero_coupon_bond_option_MC(a_param, maturity_bond, 
                                                  maturity_option, 
                                                  n_annual_trading_days, 
                                                  option_type, 
                                                  one_factor_HW_model.r_t_paths, 
                                                  sigma, strike_price, time_t, 
                                                  one_factor_HW_model.time_0_f_curve, 
                                                  one_factor_HW_model.time_0_P_curve, 
                                                  units_basis_points, None, 
                                                  one_factor_HW_model.x_t_paths)

print(f'Mean exact price: {np.mean(exact_price)}')
print(f'Mean MC function price: {np.mean(MC_price_func)}')
#%% Check whether discount factors match risk-neutral expectation of P(t,T)
t = 0.
T = 2.
print('P^M(0,T):')
print(one_factor_HW_model.time_0_P_curve[int(T*n_annual_trading_days)])
print('P(t,T):')
print(np.mean(price_zero_coupon_bond(a_param, n_annual_trading_days, 
                             one_factor_HW_model.r_t_paths[int(t*n_annual_trading_days)], 
                             sigma, t, T, one_factor_HW_model.time_0_f_curve, 
                             one_factor_HW_model.time_0_P_curve,  
                             None)))
print('D(t,T):')
print(np.mean(eval_discount_factors(n_annual_trading_days, one_factor_HW_model.r_t_paths, t, T)))
#%% Interpolate zero rate
time_t = 0.
time_T = 6.5
time_T_prev = 6.
time_T_after = 7.
alpha, tau, R_tau = interpolate_zero_rate(a_param, n_annual_trading_days, 
                      one_factor_HW_model.r_t_paths, sigma, time_t, time_T, 
                      time_T_after, time_T_prev, 
                      one_factor_HW_model.time_0_f_curve, 
                      one_factor_HW_model.time_0_R_curve, 
                      one_factor_HW_model.x_t_paths, verbose=True)
print(alpha, tau, R_tau)

#%% Evolution of swap rate over time
time_frame = np.linspace(0., 5., num=20)
swap_rate_array = np.zeros((len(time_frame), one_factor_HW_model.n_paths))
for count, time in enumerate(time_frame):
    ZCB_curve = construct_zero_coupon_bonds_curve(a_param, None, 
                                                  n_annual_trading_days, False, 
                                                  one_factor_HW_model.r_t_paths, 
                                                  sigma, 15., time, 
                                                  one_factor_HW_model.init_forward_c, 
                                                  one_factor_HW_model.init_zero_bond_c, 
                                                  one_factor_HW_model.x_t_paths)
    swap_rate_array[count] = eval_swap_rate(ZCB_curve, n_annual_trading_days, 'payer', 
                               np.arange(5., 15.+1), time)
plot_time_series(None, n_annual_trading_days, True, 'Swap Rates over Time', 
                 swap_rate_array, x_label='Time $t$', x_limits=[0., 5.], 
                 y_label='Time $t$ Swap Rate')
#%% Compare short rates with zero rates
compare_time = 2.
compare_time_idx = int(compare_time*one_factor_HW_model.n_annual_trading_days)
print(f'Short rates at time {compare_time}: {np.mean(one_factor_HW_model.r_t_paths[compare_time_idx])}')

ZCBs = construct_zero_coupon_bonds_curve(one_factor_HW_model.a_param, 
                                         None, one_factor_HW_model.n_annual_trading_days, 
                                         False, one_factor_HW_model.r_t_paths, 
                                         one_factor_HW_model.sigma, r_simulation_time, 
                                         compare_time, one_factor_HW_model.init_forward_c, 
                                         one_factor_HW_model.init_zero_bond_c, 
                                         one_factor_HW_model.x_t_paths)
zero_rates_maturity = 2.00001
zero_rates = eval_cont_comp_spot_rate(ZCBs, one_factor_HW_model.n_annual_trading_days, 
                                      compare_time, zero_rates_maturity)
print(f'Zero rates at time {compare_time} with maturity at {zero_rates_maturity}: {np.mean(zero_rates)}')

#%% Price swap, European swaption, and Bermudan LSM swaption
degree = 2
fixed_rate = 0.1
notional = 1.
payoff_var = 'swap'
plot = False
plot_regression = False
plot_timeline = False
regression_series = 'power'
moneyness = 1.
swap_type = swaption_type = 'receiver'
tenor_structure = np.arange(1., 6.+1)
time_t = 0.
units_basis_points = False
verbose = False

# Initialize classes for the swap, European swaption, and Bermudan LSM swaption
IRS = Swap(one_factor_HW_model)
European_swaption = EuropeanSwaption(fixed_rate, one_factor_HW_model, moneyness, 
                                     notional, swaption_type, tenor_structure, 
                                     time_t, units_basis_points)
Bermudan_swaption_LSM = BermudanSwaptionLSM(one_factor_HW_model)

# Evaluate the swap and swaption prices
IRS.price_forward_start_swap(fixed_rate, moneyness, notional, swap_type, 
                             tenor_structure, time_t, units_basis_points, 
                             payoff_var, plot_timeline, verbose)
print('SWAP FUNCTION TERMINATED.')
European_swaption.price_European_swaption_MC_Q(payoff_var, plot_timeline, 
                                                verbose)
print('SWAPTION MC FUNCTION TERMINATED.')
European_swaption.price_European_swaption_exact_Jamshidian(verbose)
print('SWAPTION JAMSHIDIAN FUNCTION TERMINATED.')
European_swaption.price_European_swaption_Bachelier(verbose)
print('SWAPTION BACHELIER FUNCTION TERMINATED.')
Bermudan_swaption_LSM.price_Bermudan_swaption_LSM_Q(degree, fixed_rate, 
                                                    moneyness, notional, 
                                                    swaption_type, 
                                                    tenor_structure, time_t, 
                                                    units_basis_points, payoff_var)
print('BERMUDAN LSM FUNCTION TERMINATED.')
Bermudan_swaption_LSM.eval_most_expensive_European(verbose)

print(f'tenor_structure: {tenor_structure}')
print(f'{swap_type.capitalize()} swap price: {IRS.mean_Monte_Carlo_price}')
print(f'{swaption_type.capitalize()} swaption MC price, standard error:' 
      + f' {European_swaption.mean_Monte_Carlo_price, European_swaption.se_Monte_Carlo_price}')
print(f'{swaption_type.capitalize()} swaption exact price Jamshidian:' 
      + f' {European_swaption.exact_price_Jamshidian}')
print(f'{swaption_type.capitalize()} swaption exact price Bachelier:' 
      + f' {np.mean(European_swaption.exact_price_Bachelier)}')
print(f'{swaption_type.capitalize()} Bermudan swaption LSM price, standard error:' 
      + f' {Bermudan_swaption_LSM.mean_LSM_price, Bermudan_swaption_LSM.se_LSM_price}')
print(f'MEE gap, MEE_tenor_structure: {Bermudan_swaption_LSM.MEE_gap, Bermudan_swaption_LSM.MEE_tenor_structure}')
#%%
print(European_swaption.fixed_rate)
#%% European Swaption RLNN
n_epochs = 2000
n_hidden_units = 32
learning_rate = .0004
n_runs = 1
train_frac = 1500/one_factor_HW_model.n_paths
price_vector = np.zeros(n_runs)
se_vector = np.zeros_like(price_vector)
abs_err_vector = np.zeros_like(price_vector)
save_weights = True
seed_biases = None
seed_weights = None
    
for count, _ in enumerate(range(n_runs)):
    print(f'\nn_run = {count+1}')
    EuropeanRLNN = EuropeanSwaptionRLNN(fixed_rate, one_factor_HW_model, notional, 
                                        moneyness, swaption_type, 
                                        tenor_structure, units_basis_points=units_basis_points)
    EuropeanRLNN.replicate(None, time_t, learn_rate=learning_rate, n_epochs=n_epochs, 
                           n_hidden_nodes=n_hidden_units, seed_biases=seed_biases, 
                           seed_weights=seed_weights, save_weights=save_weights, 
                           test_fit=True, train_size=train_frac)
    EuropeanRLNN.price_direct_estimator(time_t)
    
    price_vector[count] = EuropeanRLNN.mean_direct_price_estimator
    se_vector[count] = EuropeanRLNN.se_direct_price_estimator
    abs_err_vector[count] = EuropeanRLNN.mean_abs_err_direct_price_estimator
    print(f'price_vector: {price_vector}')
    print(f'se_vector: {se_vector}')
    print(f'abs_err_vector: {abs_err_vector}')
    
    with open(one_factor_HW_model.data_dir + 'EuropeanRLNN_price_results.txt', 
              'a') as f:
        f.write('\n' + 10*'*' + ' Parameters ' + 10*'*')
        f.write(f'\na parameter: {one_factor_HW_model.a_param}\n' 
                + f'number of paths: {one_factor_HW_model.n_paths}\n'
                + f'short rate process type: {one_factor_HW_model.short_rate_proc_type}\n' 
                + f'short rate simulation time: {one_factor_HW_model.short_rate_sim_time}\n'
                + f'sigma: {one_factor_HW_model.sigma}\n' 
                + f'short rate simulation type: {one_factor_HW_model.sim_type}\n'
                + f'strike price: {EuropeanRLNN.fixed_rate}\n'
                + f'tenor structure: {tenor_structure}\n'
                + f'train_size: {EuropeanRLNN.train_size}\n'
                + f'RLNN price_vector: {price_vector}\n'
                + 'RLNN price_vector mean and SEM:' 
                + f' {np.mean(price_vector[price_vector>0.])},' 
                + f' {st.sem(price_vector[price_vector>0.])}\n'
                + f'Mean rel. error with respect to Jamshidian: {EuropeanRLNN.mean_abs_err_direct_price_estimator}')
            
    
print(f'RLNN price_vector mean and SEM: {np.mean(price_vector), st.sem(price_vector)}')

## SHUTDOWN COMMAND!! ##
# os.system("shutdown /s /t 1")
#%% European EPE
# European_swaption.eval_exposures_Bachelier(True, one_factor_HW_model.r_t_paths, 
#                                            time_t, one_factor_HW_model.x_t_paths, 
#                                            units_basis_points=True)
EPETestRLNN = EuropeanSwaptionRLNN(fixed_rate, one_factor_HW_model, notional, 
                                    moneyness, swaption_type, 
                                    tenor_structure, units_basis_points=units_basis_points)
EPETestRLNN.__dict__ = EuropeanRLNN.__dict__.copy()

EPETestRLNN.eval_exposures('compare', time_t, eval_time_spacing='monthly', verbose=True)
#%% European RLNN Forward dVdR
EuroTestRLNN = EuropeanSwaptionRLNN(fixed_rate, one_factor_HW_model, notional, 
                                    moneyness, swaption_type, 
                                    tenor_structure, units_basis_points=units_basis_points)
EuroTestRLNN.__dict__ = EuropeanRLNN.__dict__.copy()

eval_times = np.arange(0., 5., 1/12)
EuroTestRLNN.eval_forward_sensitivities(eval_times, 'ISDA-SIMM')

#%% European Bachelier Forward dVdR
EuroTestRLNN.eval_forward_sensitivities_Bachelier(eval_times, 'ISDA-SIMM')
print(np.mean(EuroTestRLNN.forward_sensitivities))
#%%
# print(np.mean(EuroTestRLNN.dVdR_RLNN[0], axis=1))
# print(np.mean(EuroTestRLNN.dVdR_Bachelier[0], axis=1))
print(np.shape(EuroTestRLNN.dVdR_RLNN[:,7,:]))
print(len(st.sem(EuroTestRLNN.dVdR_RLNN[:,7,:], axis=1)))

#%% Plot European Forward Deltas
newTest =  EuropeanSwaptionRLNN(fixed_rate, one_factor_HW_model, notional, 
                                    moneyness, swaption_type, 
                                    tenor_structure, units_basis_points=units_basis_points)
newTest.__dict__ = EuroTestRLNN.__dict__.copy()
newTest.plot_forward_sensitivities('compare', True, False, 'ISDA-SIMM', True)
print()
#%% European bump-and-revalue
European_swaption_BnR = EuropeanSwaption(fixed_rate, one_factor_HW_model, notional, 
                                     swaption_type, moneyness, 
                                     tenor_structure, time_t, units_basis_points)
European_swaption_BnR.eval_forward_sensitivities_bump_and_revalue_ISDA_SIMM(
                                .0001, 'ISDA-SIMM 15Y', 0., True)
#%%
print(European_swaption.fixed_rate)
print(European_swaption_BnR.fixed_rate)
print(European_swaption_BnR.swaption_revalue_inst.fixed_rate)
print(European_swaption_BnR.se_forward_sensitivities)
#%% Compare unbumped and bumped curve values at bump time
t_idx = int(15*one_factor_HW_model.n_annual_trading_days)
print(European_swaption_BnR.model.init_zero_coupon_c[t_idx])
print(European_swaption_BnR.bump_model.init_zero_coupon_c[t_idx])
print(European_swaption_BnR.bump_model.init_zero_coupon_c_bumped[t_idx])
print(European_swaption_BnR.model.init_zero_bond_c[t_idx])
print(European_swaption_BnR.bump_model.init_zero_bond_c[t_idx])
print(European_swaption_BnR.bump_model.init_zero_bond_c_bumped[t_idx])
#%%
print(European_swaption_BnR.pathwise_forward_sensitivities)
#%%
print(np.mean(European_swaption.forw_Deltas_array_analytic[0], axis=1))
#%% Plot analytic European sensitivities
European_swaption.plot_forward_Deltas(units_basis_points=True)
#%% TestEuropean RLNN price_direct_estimator method after adjusting
EuroTestRLNN = EuropeanSwaptionRLNN(fixed_rate, one_factor_HW_model, notional, 
                                    moneyness, swaption_type, 
                                    tenor_structure, units_basis_points=units_basis_points)
EuroTestRLNN.__dict__ = EuropeanRLNN.__dict__.copy()

EuroTestRLNN.price_direct_estimator(time_t)
print(EuroTestRLNN.swaption_type)
print(EuroTestRLNN.direct_price_estimator)
print(np.mean(EuroTestRLNN.direct_price_estimator))
#%%
print(EuropeanRLNN.weights_hidden_unscaled)
print(EuropeanRLNN.weights_hidden)
print(EuropeanRLNN.biases_hidden)
#%% Initialize RLNN for Bermudan swaption and replicate
learning_rate = .0003
n_epochs = 4500
n_hidden_nodes = 64
n_runs = 1
train_frac = 2000/one_factor_HW_model.n_paths
price_vector = np.zeros(n_runs)
save_weights = True
seed_biases = None
seed_weights = None

for count, _ in enumerate(range(n_runs)):
    print(f'\nn_run = {count+1}')
    RLNN = BermudanSwaptionRLNN(fixed_rate, one_factor_HW_model, notional, 
                                moneyness, swaption_type, 
                                tenor_structure, units_basis_points=units_basis_points)
    RLNN.replicate(None, time_t, learn_rate=learning_rate, 
                   n_hidden_nodes=n_hidden_nodes, seed_biases=seed_biases, 
                   seed_weights=seed_weights, 
                   save_weights=save_weights, test_fit=True, 
                   train_size=train_frac)
    RLNN.price_direct_estimator(time_t)
    
    price_vector[count] = np.mean(RLNN.direct_price_estimator)
    print(f'price_vector: {price_vector}')
    
    with open(one_factor_HW_model.data_dir + 'RLNN_price_results.txt', 
              'a') as f:
        f.write('\n' + 10*'*' + ' Parameters ' + 10*'*')
        f.write(f'\na parameter: {one_factor_HW_model.a_param}\n' 
                + f'number of paths: {one_factor_HW_model.n_paths}\n'
                + f'short rate process type: {one_factor_HW_model.r_t_process_type}\n' 
                + f'short rate simulation time: {one_factor_HW_model.r_t_sim_time}\n'
                + f'sigma: {one_factor_HW_model.sigma}\n' 
                + f'short rate simulation type: {one_factor_HW_model.sim_type}\n'
                + f'strike price: {RLNN.fixed_rate}\n'
                + f'tenor structure: {tenor_structure}\n'
                + f'train_size: {RLNN.train_size}\n'
                + f'RLNNN price_vector: {price_vector}\n'
                + f'RLNN price_vector mean and SEM: {np.mean(price_vector[price_vector>0.]), st.sem(price_vector[price_vector>0.])}\n')
            
    
print(f'RLNN price_vector mean and SEM: {np.mean(price_vector), st.sem(price_vector)}')
#%%
print(np.shape(RLNN.model.x_t_paths))
#%% Eval Bermudan sensitivities
BermudanTestRLNN = BermudanSwaptionRLNN(fixed_rate, one_factor_HW_model, notional, 
                            moneyness, swaption_type, 
                            tenor_structure, units_basis_points=units_basis_points)
BermudanTestRLNN.__dict__ = RLNN.__dict__.copy()

eval_times = np.arange(0., tenor_structure[-2], 1/12)
BermudanTestRLNN.eval_forward_sensitivities(eval_times, 'ISDA-SIMM', verbose=True)
#%%
print(BermudanTestRLNN.model.n_paths)
#%% Plot Bermudan Forward sensitivities
newBermudanTest =  BermudanSwaptionRLNN(fixed_rate, one_factor_HW_model, notional, 
                            moneyness, swaption_type, 
                            tenor_structure, units_basis_points=units_basis_points)
newBermudanTest.__dict__ = BermudanTestRLNN.__dict__.copy()
newBermudanTest.plot_forward_sensitivities('RLNN', True, None, [0., tenor_structure[-2]], True)
#%% 
print(len(np.nonzero(newBermudanTest.pathwise_stopping_times[0,:])[0]))
#%%
print(BermudanTestRLNN.Delta_tenors)
print(f'Mean RLNN Delta: {np.mean(BermudanTestRLNN.dVdR[7])}')
print(f'S.E. RLNN Delta: {st.sem(BermudanTestRLNN.dVdR[7])}')
#%% Stopping times
BermudanTestRLNN = BermudanSwaptionRLNN(fixed_rate, one_factor_HW_model, notional, 
                            moneyness, swaption_type, 
                            tenor_structure, units_basis_points=units_basis_points)
BermudanTestRLNN.__dict__ = RLNN.__dict__.copy()

BermudanTestRLNN.eval_exercise_strategies()
print(BermudanTestRLNN.pathwise_stopping_times[:,:20])
print(np.shape(np.nonzero(BermudanTestRLNN.pathwise_stopping_times[0])[0]))

BermudanTestLSM = BermudanSwaptionLSM(one_factor_HW_model)
BermudanTestLSM.price_Bermudan_swaption_LSM_Q(degree, fixed_rate, 
                                                    moneyness, notional, 
                                                    swaption_type, 
                                                    tenor_structure, time_t, 
                                                    units_basis_points, payoff_var, 
                                                    r_t_paths=BermudanTestRLNN.pricing_r_t_paths, 
                                                    x_t_paths=BermudanTestRLNN.pricing_x_t_paths)
#%% 
print(np.shape(BermudanTestRLNN.pathwise_stopping_times))
print(np.shape(BermudanTestLSM.pathwise_stopping_times))

for count in range(len(tenor_structure)-1):
    print(len(np.nonzero(BermudanTestRLNN.pathwise_stopping_times[count]
          - BermudanTestLSM.pathwise_stopping_times[count])[0]))
#%% Visualize stopping times
fig, ax = plt.subplots(1, 1, figsize=(6,4))
fig.suptitle(f'{BermudanTestRLNN.tenor_structure_notation} Bermudan {swaption_type.capitalize()} ' 
             + f'Swaption {int(moneyness*100)}% Moneyness' 
             + '\nMean Exercise Strategy')
bar_offset = .2

mean_RLNN_live = np.sum(BermudanTestRLNN.live_paths_array, axis=1)
SEM_RLNN_live = 0
mean_LSM_live = np.sum(BermudanTestLSM.live_paths_array, axis=1)
SEM_LSM_live = 0
mean_RLNN_exercised = np.sum(BermudanTestRLNN.pathwise_stopping_times, axis=1)
SEM_RLNN_exercised = 0
mean_LSM_exercised = np.sum(BermudanTestLSM.pathwise_stopping_times, axis=1)
SEM_LSM_exercised = 0

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

plt.show()
#%%
print(BermudanTestRLNN.live_paths_array[3,:20])
print(BermudanTestLSM.live_paths_array[3,:20])
# x_axis = range(1000)
# plt.bar(x_axis, BermudanTestRLNN.live_paths_array[2,:1000])
# plt.bar(x_axis, Bermudan_swaption_LSM.live_paths_array[2,:1000])
#%% Evaluate EPE
BermudanTestRLNN = BermudanSwaptionRLNN(fixed_rate, one_factor_HW_model, notional, 
                            moneyness, swaption_type, 
                            tenor_structure, units_basis_points=units_basis_points)
BermudanTestRLNN.__dict__ = RLNN.__dict__.copy()
BermudanTestRLNN.eval_EPE_profile(True)
#%% Plot EPE
BermudanTestRLNN2 = BermudanSwaptionRLNN(fixed_rate, one_factor_HW_model, notional, 
                            moneyness, swaption_type, 
                            tenor_structure, units_basis_points=units_basis_points)
BermudanTestRLNN2.__dict__ = BermudanTestRLNN.__dict__.copy()
BermudanTestRLNN2.plot_EPE_profile('compare', units_basis_points=True)
#%%
print(np.shape(one_factor_HW_model.x_t_paths))
#%%
# print(BermudanTestRLNN.direct_price_estimator)
print(BermudanTestRLNN.mean_exposures[0])
#%%
print(len(np.nonzero(BermudanTestRLNN.pathwise_stopping_times[0])[0]))
print(len(np.nonzero(BermudanTestRLNN.pathwise_stopping_times[1])[0]))
print(len(np.nonzero(BermudanTestRLNN.pathwise_stopping_times[2])[0]))
#%% Visualize in-the-money paths
for count, date in enumerate(tenor_structure[:-1]):
    date_idx = int(date*one_factor_HW_model.n_annual_trading_days)
    plot_time_series(None, one_factor_HW_model.n_annual_trading_days, 
                     False, 'In-The-Money Paths', BermudanTestRLNN.pathwise_stopping_times[count]*
                     one_factor_HW_model.r_t_paths[date_idx:,BermudanTestRLNN.x_test_idxs], 
                     x_limits=[date, tenor_structure[-1]], y_label=r'$r_t$')
#%% Test price_direct_estimator method after adjusting
BermudanTestRLNN = BermudanSwaptionRLNN(fixed_rate, one_factor_HW_model, notional, 
                            moneyness, swaption_type, 
                            tenor_structure, units_basis_points=units_basis_points)
BermudanTestRLNN.__dict__ = RLNN.__dict__.copy()

time_t = 3.
BermudanTestRLNN.price_direct_estimator(time_t)
print(BermudanTestRLNN.direct_price_estimator)
print(np.mean(BermudanTestRLNN.direct_price_estimator))
#%% Price direct estimator
RLNN.price_direct_estimator(time_t)
print('Direct estimator, standard error:' 
      + f' {np.mean(RLNN.direct_price_estimator), st.sem(RLNN.direct_price_estimator)}')
#%%
print(RLNN.direct_price_estimator)
#%%
print(RLNN.ZBO_portfolios_weights, RLNN.biases_hidden)
#%%
print(np.sum(np.mean(one_factor_HW_model.r_t_paths[-1], axis=0)))
#%%
RLNN.neural_network.summary()

#%% Read input files as e.g.:
    # input_short_rates_100000 = pd.read_csv(input_dir+'folder\\filename.extension', header=None).to_numpy()
    
# input_short_rates_100000 = read_data(input_dir
#                                        +'100,000 1F-HW Paths\\Data\\1F-HW_Short_Rate_Paths-2022-03-18_073736.parquet')
# print(input_short_rates_100000.shape)

#%% Run RLNN for different F^M(0,t), tenors. and strikes

n_annual_trading_days = 365
simulation_time = 30.
notional = 1.
payoff_var = 'swap'
plot = False
plot_regression = False
plot_timeline = False
regression_series = 'Laguerre'
moneyness = None
swap_type = swaption_type = 'receiver'
save_weights = True
time_t = 3.
units_basis_points = False
verbose = False

# Initialize one-factor Hull-White model for given time zero rate
for time_0_rate in [.001, .03]:
    print(f'time_0_rate: {time_0_rate}')
    exp_dir_name = 'RLNN Weights for Various f(0,t), strikes, and tenor structures'
    subdir_name = f'\\F^M(0,t)={time_0_rate}'
    HW_model = OneFactorHullWhiteModel(exp_dir_name+subdir_name, 
                                       n_annual_trading_days, simulation_time, 
                                       time_0_rate)
    HW_model.construct_curves(True, 'flat', 0.)
    
    # Simulate short rates
    a_param = .01
    antithetic = True
    n_paths = 20000
    r_simulation_time = 10.
    seed = 0
    sigma = .01
    HW_model.sim_short_rate_paths(a_param, antithetic, n_paths, seed, 
                                  'zero-mean', r_simulation_time, sigma, 
                                  'Euler', time_0_rate)
    
    # Iterate over different tenors
    for tenor_structure in [[1., 2., 3., 4., 5., 6.], 
                            [5., 6., 7., 8., 9., 10.]]:
        print(f'tenor_structure: {tenor_structure}')
        
        for strike in [.1, .03, .01, .003, .001]:
            print(f'strike: {strike}')
            
            train_frac = 2000/HW_model.n_paths
            
            RLNN = BermudanSwaptionRLNN(strike, HW_model, notional, 
                                    swaption_type, tenor_structure)
            RLNN.replicate(None, 0., train_size=train_frac, n_hidden_nodes=64, 
                           learn_rate=.0003, save_weights=save_weights)
            RLNN.price_direct_estimator(time_t)
            price_vector[count] = np.mean(RLNN.direct_price_estimator)
#%%
# print(RLNN.n_hidden_nodes)            
#%%
# fixed_rate = .001
# tenor_structure = [5., 6., 7., 8., 9., 10.]
# Bermudan_swaption_LSM = BermudanSwaptionLSM(HW_model)
# Bermudan_swaption_LSM.price_Bermudan_swaption_LSM_Q(degree, fixed_rate, 
#                                                     notional, payoff_var, 
#                                                     plot, plot_regression, 
#                                                     plot_timeline, 
#                                                     regression_series, 
#                                                     moneyness, 
#                                                     swaption_type, 
#                                                     tenor_structure, time_t, 
#                                                     units_basis_points, verbose)
# Bermudan_swaption_LSM.eval_most_expensive_European(verbose)
# print(f'{swaption_type.capitalize()} Bermudan swaption LSM price, standard error:' 
#       + f' {Bermudan_swaption_LSM.mean_LSM_price, Bermudan_swaption_LSM.standard_err_price}')
# print(f'Most expensive European: {Bermudan_swaption_LSM.MEE_gap}')

# print(f'price_vector mean and SEM: {np.mean(price_vector), st.sem(price_vector)}')

#%% Shutdown script
os.system("shutdown /s /t 1")