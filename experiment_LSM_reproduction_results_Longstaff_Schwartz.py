# -*- coding: utf-8 -*-

# References:
#   [1] Longstaff, F. A., & Schwartz, E. S. (2001). Valuing American Options 
#   by Simulation: A Simple Least-Squares Approach. Review of Financial 
#   Studies, 14 (1). Retrieved from 
#   https://econpapers.repec.org/RePEc:oup:rfinst:v:14:y:2001:i:1:p:113-47

#%% Imports
from Black_Scholes_model import (generate_stock_paths_exact_cumsum, 
                                 price_European_stock_option_exact)
from datetime import datetime
from least_squares_Monte_Carlo import price_Bermudan_stock_option_LSM
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
import os
root_dir = os.chdir(os.path.dirname(os.getcwd()))

#%% Plotting style parameters
plt.style.use('fivethirtyeight')
plt.rcParams["figure.dpi"] = 150
plt.rcParams["figure.figsize"] = [6, 4]
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 16
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#%% PROGRAM
# Store program starting time for timing purposes
os.chdir(os.getcwd())
program_starting_time = datetime.now()
print(f'Program started at {program_starting_time}.')

# Assign directories for script, results, inputs, and experiments
# script_dir = os.path.dirname(__file__)
script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, 'Results\\')
input_dir = os.path.join(script_dir, 'Inputs\\')

if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

experiment_dir = os.path.join(results_dir, f'LSM Reproduction {program_starting_time.strftime("%Y-%m-%d_%H%M")}')

if not os.path.isdir(experiment_dir):
    os.makedirs(experiment_dir)
    
data_dir = os.path.join(experiment_dir, 'Data\\')

if not os.path.isdir(data_dir):
    os.makedirs(data_dir)

#%% Reproduction of Results in Section 1. A Numerical Example from ref. [1]
stock_paths = np.array([[1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
                [1.09, 1.16, 1.22, .93, 1.11, .76, .92, .88],
                [1.08, 1.26, 1.07, .97, 1.56, .77, .84, 1.22],
                [1.34, 1.54, 1.03, .92, 1.52, .90, 1.01, 1.34]], np.float64)

# Initialize stock and option parameters
degree = 2
n_annual_exercise_dates = stock_paths.shape[0] - 1
n_paths = stock_paths.shape[1]
option_type = 'put'
r = .06
regression_series = 'power'
S_0 = 1.
sigma = .1408 # Note: not defined but this value yields the right European value
simulation_time = 3.
strike_price = 1.10

# Value the put option
price_Bermudan_stock_option_LSM(degree, None, stock_paths, 
                                n_annual_exercise_dates, n_paths, option_type, 
                                False, False, r, regression_series, S_0, sigma, 
                                simulation_time, strike_price)

#%% Reproduction of Results Section in 3. Valuing American Put Options from ref. 
# [1]
n_annual_exercise_dates = 50
day_count_convention = n_annual_exercise_dates # <- not specified in LSM paper
n_paths = 100000
r = .06
regression_series = 'LaGuerre'
option_type = 'put'
ordered_put_values_array = np.zeros(20)
ordered_SE_array = np.zeros(20)
strike_price = 40
degree = 2
routine = 0

counter_var = 0
for T in [1., 2.]:
    for sigma in [.2, .4]:
        for S_0 in [36, 38, 40, 42, 44]:
            GBM = generate_stock_paths_exact_cumsum('Q', None, 
                                            n_annual_exercise_dates, n_paths, 
                                            False, r, S_0, sigma, T)
            Euro_val = price_European_stock_option_exact(T, option_type, r, 
                                                         S_0, strike_price, 
                                                         sigma, t=0)
            print(f'European option value: {Euro_val}')
            
            
            
            print('\nStock Option With Parameters as for Put Option in Longstaff-Schwartz (Section 3)')
            put = price_Bermudan_stock_option_LSM(degree, experiment_dir, GBM, 
                                                  n_annual_exercise_dates, 
                                                  n_paths, option_type, False, 
                                                  False, r, regression_series, 
                                                  S_0, sigma, T, strike_price)
            
            ordered_put_values_array[counter_var] = put[0]
            ordered_SE_array[counter_var] = put[1]
            counter_var += 1
            
#%% Plot bar charts of the results

# Longstaff & Schwartz values
values_LSM_T1_sigma20 = np.array([4.472, 3.244, 2.313, 1.617, 1.118])
values_LSM_T1_sigma40 = np.array([7.091, 6.139, 5.308, 4.588, 3.957])
values_LSM_T2_sigma20 = np.array([4.821, 3.735, 2.879, 2.206, 1.675])
values_LSM_T2_sigma40 = np.array([8.488, 7.669, 6.921, 6.243, 5.622])

errors_LSM_T1_sigma20 = np.array([.010, .009, .009, .007, .007])
errors_LSM_T1_sigma40 = np.array([.020, .019, .018, .017, .017])
errors_LSM_T2_sigma20 = np.array([.012, .011, .010, .010, .009])
errors_LSM_T2_sigma40 = np.array([.024, .022, .022, .021, .021])

values_BS_T1_sigma20 = np.array([3.844, 2.852, 2.066, 1.465, 1.017])
values_BS_T1_sigma40 = np.array([6.711, 5.834, 5.060, 4.379, 3.783])
values_BS_T2_sigma20 = np.array([3.763, 2.991, 2.356, 1.841, 1.429])
values_BS_T2_sigma40 = np.array([7.700, 6.979, 6.326, 5.736, 5.202])

# Replicated results
values_reproduction_T1_sigma20 = ordered_put_values_array[:5]
values_reproduction_T1_sigma40 = ordered_put_values_array[5:10]
values_reproduction_T2_sigma20 = ordered_put_values_array[10:15]
values_reproduction_T2_sigma40 = ordered_put_values_array[15:20]

errors_reproduction_T1_sigma20 = ordered_SE_array[:5]
errors_reproduction_T1_sigma40 = ordered_SE_array[5:10]
errors_reproduction_T2_sigma20 = ordered_SE_array[10:15]
errors_reproduction_T2_sigma40 = ordered_SE_array[15:20]

# Bar charts
s_0_axis = np.array([36, 38, 40, 42, 44])
fig, ax = plt.subplots(2, 2, figsize=[12, 8], sharex=True, sharey='row')
fig.supylabel('Option Value')
fig.supxlabel('Initial Stock Value $S_0$')
# fig.suptitle('LSM Prices of Vanilla Put Option')
width = 0.5 # width of bars

# T = 1, sigma = 0.20
ax[0,0].set_title('$T = 1$, $\sigma = 0.20$')
ax[0,0].bar(s_0_axis, values_LSM_T1_sigma20, yerr=10*errors_LSM_T1_sigma20, 
            capsize=5, width=width, label='Longstaff & Schwartz')
ax[0,0].bar(s_0_axis - width, values_reproduction_T1_sigma20, 
            yerr=10*errors_reproduction_T1_sigma20, capsize=5, 
            width=width, label='Reproduction')
ax[0,0].bar(s_0_axis + width, values_BS_T1_sigma20, width, label='European Black-Scholes')

# T = 2, sigma = 0.20
ax[0,1].set_title('$T = 2$, $\sigma = 0.20$')
ax[0,1].bar(s_0_axis, values_LSM_T2_sigma20, yerr=10*errors_LSM_T2_sigma20, 
            capsize=5, width=width, label='Longstaff & Schwartz')
ax[0,1].bar(s_0_axis - width, values_reproduction_T2_sigma20, 
            yerr=10*errors_reproduction_T2_sigma20, capsize=5, 
            width=width, label='Reproduction')
ax[0,1].bar(s_0_axis + width, values_BS_T2_sigma20, width, label='European Black-Scholes')

# T = 1, sigma = 0.40
ax[1,0].set_title('$T = 1$, $\sigma = 0.40$')
ax[1,0].bar(s_0_axis, values_LSM_T1_sigma40, yerr=10*errors_LSM_T1_sigma40,
            capsize=5, width=width, label='Longstaff & Schwartz')
ax[1,0].bar(s_0_axis - width, values_reproduction_T1_sigma40, 
            yerr=10*errors_reproduction_T1_sigma40, capsize=5, 
            width=width, label='Reproduction')
ax[1,0].bar(s_0_axis + width, values_BS_T1_sigma40, width, label='European Black-Scholes')

# T = 2, sigma = 0.40
ax[1,1].set_title('$T = 2$, $\sigma = 0.40$')
ax[1,1].bar(s_0_axis, values_LSM_T2_sigma40, yerr=10*errors_LSM_T2_sigma40, 
            capsize=5, width=width, label='Longstaff & Schwartz')
ax[1,1].bar(s_0_axis - width, values_reproduction_T2_sigma40, 
            yerr=10*errors_reproduction_T2_sigma40, capsize=5, 
            width=width, label='Reproduction')
ax[1,1].bar(s_0_axis + width, values_BS_T2_sigma40, width, label='European Black-Scholes')
plt.legend(loc='best')

# Save figure to PNG image with current date and time in filename
plot_title = None
if not experiment_dir == None:
    figures_dir = os.path.join(experiment_dir, 'Figures\\')
    
    if not os.path.isdir(figures_dir):
        os.makedirs(figures_dir)
    
    file_dir_and_name = str(figures_dir + 'LSM_Reproduction' + '-' 
                + datetime.now().strftime('%Y-%m-%d_%H%M%S'))
    plt.savefig(file_dir_and_name)
    print('\nPlot was saved to ' + file_dir_and_name + '.png')

plt.show()
plt.close()

#%% Print results in table
results_table = PrettyTable()
results_table.add_column('S_0', [36, 36, 36, 36, '', 38, 38, 38, 38, '', 40, 
                                 40, 40, 40, '', 42, 42, 42, 42, '', 44, 44, 44 
                                 , 44])
results_table.add_column('sigma', [.20, .20, .40, .40, '', .20, .20, .40, .40, 
                                   '', .20, .20, .40, .40, '', .20, .20, .40, 
                                   .40, '', .20, .20, .40, .40])
results_table.add_column('T', [1, 2, 1, 2, '', 1, 2, 1, 2, '', 1, 2, 1, 2, '', 
                               1, 2, 1, 2, '', 1, 2, 1, 2])
results_table.add_column('Longstaff & Schwartz Values', [values_LSM_T1_sigma20[0], 
                         values_LSM_T2_sigma20[0], values_LSM_T1_sigma40[0], 
                         values_LSM_T2_sigma40[0], '', 
                         values_LSM_T1_sigma20[1], values_LSM_T2_sigma20[1], 
                         values_LSM_T1_sigma40[1], values_LSM_T2_sigma40[1], 
                         '', 
                         values_LSM_T1_sigma20[2], values_LSM_T2_sigma20[2], 
                         values_LSM_T1_sigma40[2], values_LSM_T2_sigma40[2], 
                         '', 
                         values_LSM_T1_sigma20[3], values_LSM_T2_sigma20[3], 
                         values_LSM_T1_sigma40[3], values_LSM_T2_sigma40[3], 
                         '', 
                         values_LSM_T1_sigma20[4], values_LSM_T2_sigma20[4], 
                         values_LSM_T1_sigma40[4], values_LSM_T2_sigma40[4]])
results_table.add_column('L&S s.e.', [errors_LSM_T1_sigma20[0], 
                         errors_LSM_T2_sigma20[0], errors_LSM_T1_sigma40[0], 
                         errors_LSM_T2_sigma40[0], '', 
                         errors_LSM_T1_sigma20[1], errors_LSM_T2_sigma20[1], 
                         errors_LSM_T1_sigma40[1], errors_LSM_T2_sigma40[1], 
                         '', 
                         errors_LSM_T1_sigma20[2], errors_LSM_T2_sigma20[2], 
                         errors_LSM_T1_sigma40[2], errors_LSM_T2_sigma40[2], 
                         '', 
                         errors_LSM_T1_sigma20[3], errors_LSM_T2_sigma20[3], 
                         errors_LSM_T1_sigma40[3], errors_LSM_T2_sigma40[3], 
                         '', 
                         errors_LSM_T1_sigma20[4], errors_LSM_T2_sigma20[4], 
                         errors_LSM_T1_sigma40[4], errors_LSM_T2_sigma40[4]])
results_table.add_column('Reproduction Values', [values_reproduction_T1_sigma20[0], 
                                         values_reproduction_T2_sigma20[0], 
                                         values_reproduction_T1_sigma40[0], 
                                         values_reproduction_T2_sigma40[0], '', 
                                         values_reproduction_T1_sigma20[1], 
                                         values_reproduction_T2_sigma20[1], 
                                         values_reproduction_T1_sigma40[1], 
                                         values_reproduction_T2_sigma40[1], '', 
                                         values_reproduction_T1_sigma20[2], 
                                         values_reproduction_T2_sigma20[2], 
                                         values_reproduction_T1_sigma40[2], 
                                         values_reproduction_T2_sigma40[2], '', 
                                         values_reproduction_T1_sigma20[3], 
                                         values_reproduction_T2_sigma20[3], 
                                         values_reproduction_T1_sigma40[3], 
                                         values_reproduction_T2_sigma40[3], '', 
                                         values_reproduction_T1_sigma20[4], 
                                         values_reproduction_T2_sigma20[4], 
                                         values_reproduction_T1_sigma40[4], 
                                         values_reproduction_T2_sigma40[4]])
results_table.add_column('Reproduction s.e.', [errors_reproduction_T1_sigma20[0], 
                                         errors_reproduction_T2_sigma20[0], 
                                         errors_reproduction_T1_sigma40[0], 
                                         errors_reproduction_T2_sigma40[0], '', 
                                         errors_reproduction_T1_sigma20[1], 
                                         errors_reproduction_T2_sigma20[1], 
                                         errors_reproduction_T1_sigma40[1], 
                                         errors_reproduction_T2_sigma40[1], '', 
                                         errors_reproduction_T1_sigma20[2], 
                                         errors_reproduction_T2_sigma20[2], 
                                         errors_reproduction_T1_sigma40[2], 
                                         errors_reproduction_T2_sigma40[2], '', 
                                         errors_reproduction_T1_sigma20[3], 
                                         errors_reproduction_T2_sigma20[3], 
                                         errors_reproduction_T1_sigma40[3], 
                                         errors_reproduction_T2_sigma40[3], '', 
                                         errors_reproduction_T1_sigma20[4], 
                                         errors_reproduction_T2_sigma20[4], 
                                         errors_reproduction_T1_sigma40[4], 
                                         errors_reproduction_T2_sigma40[4]])
results_table.add_column('European Black-Scholes', [values_BS_T1_sigma20[0], 
                         values_BS_T2_sigma20[0], values_BS_T1_sigma40[0], 
                         values_BS_T2_sigma40[0], '', 
                         values_BS_T1_sigma20[1], values_BS_T2_sigma20[1], 
                         values_BS_T1_sigma40[1], values_BS_T2_sigma40[1], 
                         '', 
                         values_BS_T1_sigma20[2], values_BS_T2_sigma20[2], 
                         values_BS_T1_sigma40[2], values_BS_T2_sigma40[2], 
                         '', 
                         values_BS_T1_sigma20[3], values_BS_T2_sigma20[3], 
                         values_BS_T1_sigma40[3], values_BS_T2_sigma40[3], 
                         '', 
                         values_BS_T1_sigma20[4], values_BS_T2_sigma20[4], 
                         values_BS_T1_sigma40[4], values_BS_T2_sigma40[4]])
print(results_table)

runtime = (datetime.now() - program_starting_time)

#%% Save log file of statistical and computational details if experiment 
if experiment_dir:
    with open(data_dir+f'LSM_{option_type}_option_log.txt', 'a') as f:
        f.write(10*'*' + ' Statistical Details ' + 10*'*')
        f.write(f'\n{results_table}\n\n')
        
        f.write(10*'*' + ' Computational Details ' + 10*'*')
        f.write('\nTotal LSM Longstaff & Schwartz results reproduction' 
                + f' runtime: {runtime}\n')
        
#%% Shutdown script
# os.system("shutdown /s /t 1")