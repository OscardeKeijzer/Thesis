# -*- coding: utf-8 -*-

# Imports
from datetime import datetime
import os
import pandas as pd
import numpy as np
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
from swaptions import (price_European_swaption_exact_Jamshidian, 
                       price_European_swaption_MC_Q)

#%% Initialize 1F-HW model instance:
n_annual_trading_days = 365
simulation_time = 30.
time_0_rate = .001
one_factor_HW_model = OneFactorHullWhiteModel(None, n_annual_trading_days, 
                                                simulation_time, time_0_rate)
one_factor_HW_model.construct_curves(True, 'flat', 0.)
#%% Simulate short rates
a_param = .01
antithetic = True
n_paths = 5000
r_simulation_time = 10.
seed = 0
sigma = .01
one_factor_HW_model.sim_short_rate_paths(a_param, antithetic, n_paths, seed, 
                                         'zero-mean', r_simulation_time, sigma, 
                                         'Euler', time_0_rate)
#%% Plot short rates
one_factor_HW_model.plot_short_rate_paths(r_simulation_time, 'mean')
#%% Log short rates
one_factor_HW_model.log_short_rate_paths()
#%% Price zero-bond options
maturity_bond = 10.
maturity_option = 5.
option_type = 'call'
strike_price = .1
time_t = 0.
t_idx = int(time_t*n_annual_trading_days)
T_idx = int(maturity_option*n_annual_trading_days)
units_basis_points = False

print('\nUsing short rates:')
exact_price = price_zero_coupon_bond_option_exact(a_param, maturity_bond, 
                                                  maturity_option, 
                                                  n_annual_trading_days, 
                                                  option_type, 
                                                  one_factor_HW_model.r_t_paths, 
                                                  sigma, strike_price, time_t, 
                                                  one_factor_HW_model.init_forward_c, 
                                                  one_factor_HW_model.init_zero_bond_c, 
                                                  units_basis_points, 
                                                  None, 
                                                  None)
MC_price = price_zero_coupon_bond_option_MC(a_param, maturity_bond, 
                                                  maturity_option, 
                                                  n_annual_trading_days, 
                                                  option_type, 
                                                  one_factor_HW_model.r_t_paths, 
                                                  sigma, strike_price, time_t, 
                                                  one_factor_HW_model.init_forward_c, 
                                                  one_factor_HW_model.init_zero_bond_c, 
                                                  units_basis_points, 
                                                  None, 
                                                  one_factor_HW_model.x_t_paths)

print(f'Mean exact price: {np.mean(exact_price)}')
print(f'Mean MC price: {np.mean(MC_price)}')

# print('\nUsing zero-mean process:')
# exact_price = price_zero_coupon_bond_option_exact(a_param, maturity_bond, 
#                                                   maturity_option, 
#                                                   n_annual_trading_days, 
#                                                   option_type, 
#                                                   one_factor_HW_model.r_t_paths[t_idx], 
#                                                   sigma, strike_price, time_t, 
#                                                   one_factor_HW_model.init_forward_c, 
#                                                   one_factor_HW_model.init_zero_bond_c, 
#                                                   units_basis_points, 
#                                                   None, 
#                                                   one_factor_HW_model.x_t_paths[t_idx])
# MC_price = price_zero_coupon_bond_option_MC(a_param, maturity_bond, 
#                                                   maturity_option, 
#                                                   n_annual_trading_days, 
#                                                   option_type, 
#                                                   one_factor_HW_model.r_t_paths[t_idx], 
#                                                   sigma, strike_price, time_t, 
#                                                   one_factor_HW_model.init_forward_c, 
#                                                   one_factor_HW_model.init_zero_bond_c, 
#                                                   units_basis_points, 
#                                                   None, 
#                                                   one_factor_HW_model.x_t_paths[t_idx])

# print(f'Mean exact price: {np.mean(exact_price)}')
# print(f'Mean MC price: {np.mean(MC_price)}')
#%%
print(one_factor_HW_model.init_zero_bond_c[int(5.*n_annual_trading_days)])
print(np.mean(price_zero_coupon_bond(a_param, n_annual_trading_days, 
                             one_factor_HW_model.r_t_paths[int(0.*n_annual_trading_days)], 
                             sigma, time_t, 5., one_factor_HW_model.init_forward_c, 
                             one_factor_HW_model.init_zero_bond_c,  
                             None)))
#%% Price swap, European swaption, and Bermudan LSM swaption
degree = 2
fixed_rate = .01
notional = 1.
payoff_var = 'swap'
plot = False
plot_regression = False
plot_timeline = False
regression_series = 'Laguerre'
swap_rate_moneyness = None
swap_type = swaption_type = 'receiver'
tenor_structure = [5., 6., 8., 10.]
time_t = 0.
units_basis_points = False
verbose = False

# Initialize classes for the swap, European swaption, and Bermudan LSM swaption
IRS = Swap(one_factor_HW_model)
European_swaption = EuropeanSwaption(one_factor_HW_model)
Bermudan_swaption_LSM = BermudanSwaptionLSM(one_factor_HW_model)

# Evaluate the swap and swaption prices
IRS.price_forward_start_swap(fixed_rate, notional, payoff_var, False, 
                              swap_rate_moneyness, swap_type, tenor_structure, 
                              time_t, units_basis_points, verbose)
European_swaption.price_European_swaption_MC_Q(fixed_rate, notional, 
                                                payoff_var, plot_timeline, 
                                                swaption_type, 
                                                swap_rate_moneyness, 
                                                tenor_structure, time_t, 
                                                units_basis_points, verbose)
European_swaption.price_European_swaption_exact_Jamshidian(fixed_rate, notional, 
                                                plot_timeline, sigma, 
                                                swaption_type, 
                                                tenor_structure, time_t, 
                                                units_basis_points, 
                                                verbose)
Bermudan_swaption_LSM.price_Bermudan_swaption_LSM_Q(degree, fixed_rate, 
                                                    notional, payoff_var, 
                                                    plot, plot_regression, 
                                                    plot_timeline, 
                                                    regression_series, 
                                                    swap_rate_moneyness, 
                                                    swaption_type, 
                                                    tenor_structure, time_t, 
                                                    units_basis_points, verbose)

(print(f'{swap_type.capitalize()} swap price: {IRS.swap_price}') if time_t == 0 
 else print(f'{swap_type.capitalize()} swap price: {IRS.mean_Monte_Carlo_price}'))
print(f'{swaption_type.capitalize()} swaption MC price:' 
      + f' {European_swaption.mean_Monte_Carlo_price}')
print(f'{swaption_type.capitalize()} swaption exact price:' 
      + f' {European_swaption.exact_price}')
print(f'{swaption_type.capitalize()} Bermudan swaption LSM price:' 
      + f' {Bermudan_swaption_LSM.mean_LSM_price}')
#%% Initialize RLNN for Bermudan swaption and replicate
RLNN = BermudanSwaptionRLNN(fixed_rate, one_factor_HW_model, notional, 
                            swaption_type, tenor_structure)
RLNN.replicate(None, 0.)
# RLNN.price_direct_estimator()
#%%
print(np.mean(RLNN.pathwise_price_vector, axis=1))
#%% Use direct estimator method
RLNN.price_direct_estimator()
print(f'mean pathwise_price_vector: {np.mean(RLNN.pathwise_price_vector, axis=(1,2))}')
print(f'shape of pathwise_price_vector: {np.shape(RLNN.pathwise_price_vector)}')
#%%
RLNN.neural_network.summary()
# print(RLNN.ZBO_portfolios)

for count, monitor_date in enumerate(tenor_structure[:-1]):
    print(f'\nReplicating portfolio value at time {monitor_date}')
    print(np.mean(RLNN.tenor_discount_factors*(np.mean(RLNN.ZBO_portfolios[count], 
                                                       axis=0))))


#%% Delta sensitivity
# Bump-and-revalue
bump_size = time_0_rate/10
eval_times = [1/26, 1/12, 1/4, 1/2, 1., 2., 4., 5., 10.]
Bermudan_swaption_LSM.eval_forward_Deltas_bump_and_revalue(bump_size, 
                                                           eval_times, verbose)
print(Bermudan_swaption_LSM.mean_forw_Deltas)
#%% Read input files as e.g.:
    # input_short_rates_100000 = pd.read_csv(input_dir+'folder\\filename.extension', header=None).to_numpy()
    
# input_short_rates_100000 = read_data(input_dir
#                                        +'100,000 1F-HW Paths\\Data\\1F-HW_Short_Rate_Paths-2022-03-18_073736.parquet')
# print(input_short_rates_100000.shape)

#%% Shutdown script
# os.system("shutdown /s /t 1")