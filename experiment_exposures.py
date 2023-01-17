# -*- coding: utf-8 -*-

# Imports
from datetime import datetime
import os
import pandas as pd
import numpy as np

os.chdir(os.getcwd())

# Local imports
from bonds_and_bond_options import (construct_zero_coupon_bonds_curve, 
                                    price_coupon_bearing_bond, 
                                    price_coupon_bearing_bond_option, 
                                    price_zero_coupon_bond, 
                                    price_zero_coupon_bond_option_exact)
from classes import (BermudanSwaptionRLNN, 
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
from one_factor_Hull_White_model import (gen_one_factor_Hull_White_paths)
from plotting_functions import (plot_time_series, 
                                plot_one_factor_Hull_White_histogram)
from swaps import (eval_annuity_terms, 
                   eval_swap_rate, 
                   price_forward_start_swap)
from swaptions import (price_European_swaption_exact_Jamshidian, 
                       price_European_swaption_MC_Q)

#%% Using class for 1F-HW model:
n_annual_trading_days = 253
simulation_time = 30.
time_0_rate = .001
one_factor_HW_model = OneFactorHullWhiteModel('Exposures', n_annual_trading_days, 
                                                simulation_time, time_0_rate)
one_factor_HW_model.construct_curves(True, 'flat', 0.)
#%% # Simulate short rates
a_param = .01
n_paths = 10
r_simulation_time = 10.
sigma = .01
one_factor_HW_model.sim_short_rate_paths(False, a_param, n_paths, 'zero-mean', 
                                         r_simulation_time, sigma, 'euler', 
                                         time_0_rate)
one_factor_HW_model.plot_short_rate_paths(None, 'both')
#%% # European swaption exposures
fixed_rate = .0001
n_paths_exposure = n_paths
notional = 1.
payoff_var = 'swap'
plot_exposure = True
swaption_type = 'receiver'
swap_rate_multiplier = None
tenor_structure = [1., 2., 3., 4., 5.]
time_t = 0.
verbose = False
eval_time_spacing = 'monthly'

Euro_swaption = EuropeanSwaption(one_factor_HW_model)
Euro_swaption.eval_exposures(fixed_rate, n_paths_exposure, notional, payoff_var, 
                             plot_exposure, swaption_type, swap_rate_multiplier, 
                             tenor_structure, time_t, verbose, eval_time_spacing)
#%%
print(Euro_swaption.exposure_array[:,0])