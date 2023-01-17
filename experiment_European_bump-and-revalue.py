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
from interest_rate_functions import (eval_annuity_terms,
                                     eval_cont_comp_spot_rate,
                                     eval_swap_rate)
from least_squares_Monte_Carlo import (price_Bermudan_stock_option_LSM, 
                                       price_Bermudan_swaption_LSM_Q)
from neural_networks import (HyperParameterTunerShallowFeedForwardNeuralNetwork, 
                             ShallowFeedForwardNeuralNetwork)
from one_factor_Hull_White_model import (eval_discount_factors,
                                         gen_one_factor_Hull_White_paths)
from plotting_functions import (plot_time_series, 
                                plot_one_factor_Hull_White_histogram)
from swaps import (price_forward_start_swap)
from swaptions import (price_European_swaption_exact_Jamshidian, 
                       price_European_swaption_MC_Q)
import tensorflow.keras as keras



#%% Bump-and-revalue simulation routine

_n_seeds = 5

for seed in range(0,_n_seeds):
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
    from interest_rate_functions import (eval_annuity_terms,
                                         eval_cont_comp_spot_rate,
                                         eval_swap_rate)
    from least_squares_Monte_Carlo import (price_Bermudan_stock_option_LSM, 
                                           price_Bermudan_swaption_LSM_Q)
    from neural_networks import (HyperParameterTunerShallowFeedForwardNeuralNetwork, 
                                 ShallowFeedForwardNeuralNetwork)
    from one_factor_Hull_White_model import (eval_discount_factors,
                                             gen_one_factor_Hull_White_paths)
    from plotting_functions import (plot_time_series, 
                                    plot_one_factor_Hull_White_histogram)
    from swaps import (price_forward_start_swap)
    from swaptions import (price_European_swaption_exact_Jamshidian, 
                           price_European_swaption_MC_Q)
    import tensorflow.keras as keras
    
    # 1F-HW model parameters
    exp_dir_name = 'European Bump-and-Revalue 5Yx10Y Time-Zero 15Y Delta TESTING'
    n_annual_trading_days = 253
    simulation_time = 30.
    time_0_rate = .001
    
    # Swaption parameters
    fixed_rate = .001
    notional = 1.
    payoff_var = 'swap'
    plot = False
    plot_timeline = False
    swap_rate_moneyness = 1.
    swaption_type = 'receiver'
    tenor_structure = np.arange(5., 15.+1)
    time_t = 0.
    units_basis_points = True
    verbose = False

    # Short rate simulation parameters
    a_param = .01
    antithetic = True
    n_paths = 50000
    r_simulation_time = tenor_structure[-1]
    sigma = .01

    # Bump-and-revalue parameters
    bump_size = .0001
    bump_time = 'ISDA-SIMM 15Y'
    eval_time = 0.
    plot_bumped_curve = True

    mean_delta_vector = np.zeros(_n_seeds)
    se_delta_vector = np.zeros_like(mean_delta_vector)
    timing_vector = np.zeros_like(mean_delta_vector)
    
    count = seed
    print(f'run: {count+1}')
    
    # Instantiate 1F-HW model, simulate short rates, then bump-and-revalue 
    # and save the results
    one_factor_HW_model = OneFactorHullWhiteModel(exp_dir_name, n_annual_trading_days, 
                                                  simulation_time, time_0_rate)
    one_factor_HW_model.construct_curves(True, 'flat', 'continuously-compounded', 
                                         0., False)
    
    one_factor_HW_model.sim_short_rate_paths(a_param, antithetic, 
                                             one_factor_HW_model.init_forward_c, 
                                             one_factor_HW_model.init_forward_c_dfdT, 
                                             n_paths, seed, 'zero-mean', 
                                             r_simulation_time, sigma, 'Euler', 
                                             time_0_rate)
    
    European_swaption = EuropeanSwaption(fixed_rate, one_factor_HW_model, 
                                         notional, swaption_type, 
                                         swap_rate_moneyness, tenor_structure, 
                                         time_t, units_basis_points)
    European_swaption.price_European_swaption_MC_Q('swap', False, False)

    BnR_starting_time = datetime.now()
    European_swaption.eval_forward_sensitivities_bump_and_revalue_ISDA_SIMM(bump_size, 
                                                        bump_time, eval_time, 
                                                        plot_bumped_curve, 
                                                        seed)
    BnR_finish_time = datetime.now()
    
    with open(one_factor_HW_model.data_dir + 'European_BnR_Delta_Means.txt', 
              'a') as f:
        f.write(f'{European_swaption.mean_forward_sensitivity}, ')
        
    with open(one_factor_HW_model.data_dir + 'European_LBnR_Delta_SEs.txt', 
              'a') as f:
        f.write(f'{European_swaption.se_forward_sensitivities}, ')
        
    with open(one_factor_HW_model.data_dir + 'European_BnR_Timings.txt', 
              'a') as f:
        f.write(f'{(BnR_finish_time - BnR_starting_time).total_seconds()}, ')
        
    # Clear all global variables from memory to prevent RAM depletion
    for name in dir():
        if not name.startswith('_'):
            del globals()[name]
#%%
# np.savetxt(one_factor_HW_model.data_dir+'mean_delta_vector.txt', 
#            mean_delta_vector, delimiter=', ', fmt='%f', 
#            header=f'Mean time-zero 15Y Deltas out of {n_paths} pathwise values each')
# np.savetxt(one_factor_HW_model.data_dir+'se_delta_vector.txt', 
#            se_delta_vector, delimiter=', ', fmt='%f', 
#            header=f'SE of time-zero 15Y Deltas out of {n_paths} pathwise values each')
# np.savetxt(one_factor_HW_model.data_dir+'timing_vector.txt', 
#            timing_vector, delimiter=', ', fmt='%f', 
#            header=f'Runtime in seconds of bump-and-revalue method for time-zero 15Y Deltas out of {n_paths} pathwise values')

#%%
## SHUTDOWN COMMAND!! ##
# import os
# os.system("shutdown /s /t 1")