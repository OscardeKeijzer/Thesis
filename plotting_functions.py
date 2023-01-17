# -*- coding: utf-8 -*-

# Imports
from datetime import datetime
from keras_visualizer import visualizer
import math
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
os.chdir(os.getcwd())
import pandas as pd
import scipy.stats as st
import warnings
warnings.simplefilter('ignore', np.RankWarning)

# Plotting style parameters
plt.style.use('fivethirtyeight')
plt.rcParams["figure.dpi"] = 150
plt.rcParams["figure.figsize"] = [6, 4]
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 14
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Function definitions
def curly_arrow(ax, 
                start, 
                end, 
                col, 
                arrow_type, 
                ):
    """
    Info: Adds a possibly curly arrow as a marker to a plot. 
        Credit to hayk (https://stackoverflow.com/users/4888158/hayk)
        for his post providing the framework of this function 
        at https://stackoverflow.com/questions/45365158/matplotlib-wavy-arrow
        
    Input: 
        
    Output: 
        
    """
    """
    TO-DO:  -Update doc strings
    """
    fig_width, fig_height = plt.gcf().get_size_inches()*plt.gcf().dpi
    xlim0, xlim1 = ax.get_xlim()
    ylim0, ylim1 = ax.get_ylim()
    triangle_height = ylim1*10**-1 #arr_size/1000#fig_height*10**-5
    triangle_width = xlim1*10**-2 #arr_size/1000#fig_width*10**-4/2
    width = 1/3*triangle_width
    linew=2.
    
    # Wiggly line
    xmin, ymin = start
    xmax, ymax = end
    dist = np.sqrt((xmin - xmax)**2 + (ymin - ymax)**2) - triangle_height
    n0 = dist / (2 * np.pi)
    if arrow_type.lower() == 'straight':
        n = 0
    else:
        n = 4#ymax/0.01
    
    x = np.linspace(0, dist, 151) + xmin
    y = width * np.sin(n * x / n0) + ymin
    line = plt.Line2D(x,y, color=col, lw=linew)
    
    del_x = xmax - xmin
    del_y = ymax - ymin
    ang = np.arctan2(del_y, del_x)
    
    line.set_transform(mpl.transforms.Affine2D().rotate_around(xmin, ymin, ang) + ax.transData)
    ax.add_line(line)

    # Triangle
    verts = np.array([[-triangle_height,0],[-triangle_height,-triangle_width],
                      [0,0],[-triangle_height,triangle_width]]).astype(float)
    verts[:,1] += ymax
    verts[:,0] += xmax
    path = mpath.Path(verts)
    patch = mpatches.PathPatch(path, fc=col, ec=col)

    patch.set_transform(mpl.transforms.Affine2D().rotate_around(xmax, ymax, ang) + ax.transData)
    
    return patch

def plot_time_series(experiment_dir: str,
                     n_annual_trading_days: int,
                     plot_mean: bool,
                     plot_title: str,
                     time_series: list,
                     n_x_ticks: int = 6,
                     x_label: str = None,
                     x_limits: list = None,
                     y_label: str = None,
                     y_limits: list = None
                    ) -> None:
    """
    Info:
        This function takes a possibly multidimensional time series and plots 
            either all subseries or the mean of the time series.
        
    Input:
        experiment_dir: the directory to which the results are saved.
        
        n_annual_trading_days: the number of trading days in a year. If float, the
            x axis is plotted in years; if None, the length of the time series is
            plotted on the x axis.
        
        plot_mean: if True, the mean of the multidimensional time series is 
            plotted; if False, the individual subseries are plotted.
        
        plot_title: the string that is to be displayed as the title of the 
            output plot.
        
        time_series: a 1D (or 2D) array containing one (or more) asset price 
            time series.
        
        x_label: the label to be displayed on the x-axis.
        
        x_limits: a list with the first and second elements being the lower and 
            upper limits, respectively, for the x-axis.
        
        y_label: the label to be displayed on the y-axis.
        
        y_limits: a list with the first and second elements being the lower and 
        upper limits, respectively, for the y-axis.
        
    Output:
        A plot of the mean or individual time series values over time.
    """
    """
    TO-DO: When x axis in years, round tick values to integers
    """
    # Check whether Figures directory located in current directory and if not, create Figures directory
    if not experiment_dir == None:
        figures_dir = os.path.join(experiment_dir, 'Figures\\')
    
        if not os.path.isdir(figures_dir):
            os.makedirs(figures_dir)

    # Initialize figure
    fig, ax = plt.subplots()
    
    # Adjust axis limits if passed
    if x_limits:
        ax.set_xlim(x_limits[0], x_limits[1])
    else:
        ax.set_xlim(0, time_series.shape[0])
        
    if not y_limits == None:
        plt.ylim(y_limits[0], y_limits[1])
        
    # Adjust number of ticks if passed to prevent overlap
    if n_x_ticks is not None:
        if n_x_ticks > 10:
            n_x_ticks = math.ceil(n_x_ticks/4.)*2
        if n_x_ticks > 20:
            n_x_ticks = math.ceil(n_x_ticks/8.)*2
    else:
        n_x_ticks = 6
    
    if plot_mean == True:
        if len(time_series.shape) > 1:
            # Plot mean of time series
            plt.suptitle(f'Mean of {time_series.shape[1]:,} ' + plot_title)
            if x_label == None:
                ax.set_xlabel('Time $t$')
            ax.set_ylabel(y_label)
            
            # If day count convention was passed, plot the x axis in years
            if not n_annual_trading_days == None:
                # ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                xticks_locs = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 
                                          n_x_ticks)
                if x_limits:
                    xticks_labels = np.linspace(ax.get_xlim()[0], 
                                                ax.get_xlim()[1], n_x_ticks)
                else:
                    xticks_labels = np.linspace(ax.get_xlim()[0], 
                                                int(ax.get_xlim()[1]
                                                    /n_annual_trading_days), 
                                                n_x_ticks)
                    
                ax.set_xticks(xticks_locs, ["%.1f" % x for x in xticks_labels])
                if not x_label == None:
                    ax.set_xlabel(x_label + ' (years)')
                
            time_series_mean = np.mean(time_series, axis=1)
            time_series_std_err = st.sem(time_series, axis=1)
            x_axis = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 
                                 time_series.shape[0])
            ax.plot(x_axis, time_series_mean, color='black', linewidth=1)
            
            # Plot standard error if values greater than machine epsilon
            if not np.all(time_series_std_err < np.finfo(float).eps):
                ax.fill_between(x_axis, time_series_mean-time_series_std_err, 
                             time_series_mean+time_series_std_err, alpha=0.5, 
                             label='Standard Error')
            ax.legend(loc='best')
            plt.tight_layout()
            
            # Save figure to PNG image with current date and time in filename
            if not experiment_dir == None:
                file_dir_and_name = str(figures_dir + 'Mean_of_' 
                                        + ((plot_title.replace(' ','_')
                                            ).replace('\n','_')
                                           ).replace('\mathbb{Q}', 'Q')  + '-' 
                            + datetime.now().strftime('%Y-%m-%d_%H%M%S'))
                plt.savefig(file_dir_and_name)
                print('\nPlot was saved to ' + file_dir_and_name + '.png')
        else:
            raise TypeError('cannot plot mean of a single run')
    else:
        # Plot individual time series
        if time_series.ndim > 1:
            plt.suptitle(f'{time_series.shape[1]:,} ' + plot_title)
            if x_label == None:
                ax.set_xlabel('Time $t$')
            ax.set_ylabel(y_label)
            
            # If day count convention was passed, plot the x axis in years
            if not n_annual_trading_days == None:
                # ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                xticks_locs = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], n_x_ticks)
                if x_limits:
                    xticks_labels = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 
                                                n_x_ticks)
                else:
                    xticks_labels = np.linspace(ax.get_xlim()[0], 
                                                int(ax.get_xlim()[1]
                                                    /n_annual_trading_days), n_x_ticks)
                    
                ax.set_xticks(xticks_locs, ["%.1f" % x for x in xticks_labels])
                if not x_label == None:
                    ax.set_xlabel(x_label + ' (years)')
                
            x_axis = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 
                                 time_series.shape[0])
            for series in range(time_series.shape[1]):
                ax.plot(x_axis, time_series[:,series], linewidth=1)
            
            plt.tight_layout()
            
        else:
            plt.suptitle(plot_title)
            if x_label == None:
                ax.set_xlabel('Time $t$')
            ax.set_ylabel(y_label)
            
            # If day count convention was passed, plot the x axis in years
            if not n_annual_trading_days == None:
                # ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                xticks_locs = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 
                                          n_x_ticks)
                if x_limits:
                    xticks_labels = np.linspace(ax.get_xlim()[0],
                                                ax.get_xlim()[1], n_x_ticks)
                else:
                    xticks_labels = np.linspace(ax.get_xlim()[0], 
                                                int(ax.get_xlim()[1]
                                                    /n_annual_trading_days), n_x_ticks)
                    
                ax.set_xticks(xticks_locs, ["%.1f" % x for x in xticks_labels])
                if not x_label == None:
                    ax.set_xlabel(x_label + ' (years)')
            
            x_axis = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 
                                 time_series.shape[0])
            ax.plot(x_axis, time_series, linewidth=1)
            plt.tight_layout()
        
        # Save figure to PNG image with current date and time in filename
        if not experiment_dir == None:
            file_dir_and_name = str(figures_dir 
                                    + ((plot_title.replace(' ','_')
                                        ).replace('\n','_')
                                       ).replace('\mathbb{Q}', 'Q') + '-' 
                                    + datetime.now().strftime('%Y-%m-%d_%H%M%S'))
            plt.savefig(file_dir_and_name)
            print('\nPlot was saved to ' + file_dir_and_name + '.png')
    
    
    # Display and then close figure
    plt.show()
    plt.close()
    
def plot_one_factor_Hull_White_histogram(experiment_dir: str,
                                         expectation: float,
                                         n_annual_trading_days: int,
                                         plot_title: str,
                                         time_point: float,
                                         short_rate_paths: list,
                                         variance: float,
                                         x_label: str
                                        ) -> None:
    """
    Info:
        This plots a histogram of the simulated one-factor Hull-White (1F-HW) 
        short rate path values at a specified time point with an overlaid 
        outline of the expected distribution.
        
    Input:
        n_annual_trading_days: the number of trading days in a year. If float, the
            x axis is plotted in years; if None, the length of the time series is
            plotted on the x axis.
        
        experiment_dir: the directory to which the results are saved.
        
        expectation: the expectation of the 1F-HW short rates at time t conditional on
            the values at time s.
        
        plot_title: the string that is to be displayed as the title of the output plot.
        
        time_point: the time point in years of the input time series at which 
            the histogram is evaluated.
            
        short_rate_paths: a 2D array containing the 1F-HW short rate paths.
            
        variance: the variance of the 1F-HW short rates at time t conditional
            on the values at time s.
        
        x_label: the label to be displayed on the x-axis.
        
    Output:
        ...
    """
    # Check whether Figures directory located in current directory and if not, create Figures directory
    if not experiment_dir == None:
        figures_dir = os.path.join(experiment_dir, 'Figures\\')
    
        if not os.path.isdir(figures_dir):
            os.makedirs(figures_dir)
    
    # Create pandas DataFrame of mean 1F-HW short rate data at specified time point
    df = pd.DataFrame(short_rate_paths[int(time_point*n_annual_trading_days)])
    std_dev = np.sqrt(variance)
    total_time = short_rate_paths.shape[0]-1
    
    # Initialize figure
    fig, ax = plt.subplots()
    
    # Plot histogram of time series
    ax.set(xlabel=x_label, ylabel='Frequency')
    
    # Plot histogram of 1F-HW short rates and store bins
    hist, bins, _ = ax.hist(df, bins='auto')
    # evaluate normal probability distribution function given normalized log return data and bins
    norm_pdf = st.norm.pdf(bins, expectation, std_dev)
    # overlay normal probability distribution function
    ax.plot(bins, norm_pdf*hist.sum()/norm_pdf.sum(), color='black', 
            linestyle='--', linewidth=2, label='Analytical Distribution')
    ax.axvline(expectation, color='C01', linewidth=2, label='Expectation')
    # ax.text(expectation, -.05, f'{expectation:.5f}', fontsize=MEDIUM_SIZE, color='black', 
    #         transform=ax.get_xaxis_transform(), ha='center', va='top')
    if n_annual_trading_days:
        plt.suptitle(f'Histogram at {time_point} Years of' 
                     + f' {short_rate_paths.shape[1]:,}\n' + plot_title)
    else:
        plt.suptitle(f'Histogram at Time {time_point:,}/{total_time:,} of' 
                     + f' {short_rate_paths.shape[1]:,} ' + plot_title)
    plt.legend(loc='best')
    plt.tight_layout()
    
    # Save figure to PNG image with current date and time in filename
    if not experiment_dir == None:
        file_dir_and_name = str(figures_dir +  
                                ((plot_title.replace(' ','_')
                                    ).replace('\n','_')
                                   ).replace('\mathbb{Q}', 'Q') + '_Histogram' 
                                + '-' + datetime.now().strftime('%Y-%m-%d_%H%M%S'))
        plt.savefig(file_dir_and_name)
        print('\nHistogram was saved to ' + file_dir_and_name + '.png')
    
    # Delete DataFrame to clear memory
    del df
    
def plot_forward_start_swap_timeline(fixed_rate: float,
                                     floating_rates: list,
                                     swap_type: str,
                                     tenor_structure: list
                                    ) -> None:
    """
    Info: This function plots the timeline of a forward swap in order to 
        illustrate the tenor.
    
    Input:
        fixed_rate: the fixed interest rate of the forward swap.
        
        floating_rates: a list containing the realized interest rate values at
            the reset dates.
        
        swap_type: whether the swap is a payer or receiver forward swap.
        
        tenor_structure: a list containing the forward swap starting date as 
            the first entry and the payment dates as the remaining entries.
            
    Output:
        A plot of the forward swap timeline.
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_xlim(0, tenor_structure[-1] + 1/10)
    
    # Adjust y axis limits and size of arrows
    if fixed_rate >= np.max(floating_rates):
        ax.set_ylim(-1.1*fixed_rate, 1.1*fixed_rate)
    else:
        ax.set_ylim(-1.1*np.max(floating_rates), 1.1*np.max(floating_rates))
    
    # Configure main x axis
    xticks_locs = tenor_structure
    xticks_labels = np.array([r'$T_0$' + '$_+$' + f'$_{idx}$' 
                              for idx in range(1,len(tenor_structure)-1)], 
                             dtype=str)
    xticks_labels = np.concatenate((r'$T_0$', xticks_labels), axis=None)
    xticks_labels = np.concatenate((xticks_labels, r'$T_M$'), axis=None)
    ax.set_xticks(xticks_locs, xticks_labels)
    ax.grid(True, which='both')
    
    # Configure title and y axis
    plt.suptitle(f'{swap_type.capitalize()} Forward-Start Interest Rate Swap Timeline')
    ax.set_ylabel('Interest Rate')
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.axhline(y=0, color='gray', lw=2)
    fig.autofmt_xdate(ha='center')
    
    # Plot arrows at payment dates 
    if swap_type.lower() == 'payer':
        for count, T_payment in enumerate(tenor_structure[1:]):
            ax.add_patch(curly_arrow(ax, (T_payment, 0), (T_payment, floating_rates[count]), 'C0', arrow_type='curly'))
            ax.add_patch(curly_arrow(ax, (T_payment, 0), (T_payment, -fixed_rate), 'C1', arrow_type='straight'))
            ax.scatter(tenor_structure[:-1], np.zeros(len(tenor_structure)-1), color='C2', marker='o', s=225)
            
            plt.legend(handles=[mpatches.Patch(color='C0', label='Floating Payment'),
                                mpatches.Patch(color='C1', label='Fixed Payment'),
                                mpatches.Patch(color='C2', label='Fixing Date')], loc='best')
    elif swap_type.lower() == 'receiver':
        for count, T_payment in enumerate(tenor_structure[1:]):
            ax.add_patch(curly_arrow(ax, (T_payment, 0), (T_payment, fixed_rate), 'C0', arrow_type='straight'))
            ax.add_patch(curly_arrow(ax, (T_payment, 0), (T_payment, -floating_rates[count]), 'C1', arrow_type='curly'))
            ax.scatter(tenor_structure[:-1], np.zeros(len(tenor_structure)-1), color='C2', marker='o', s=225)
            plt.legend(handles=[mpatches.Patch(color='C0', label='Fixed Payment'), 
                                mpatches.Patch(color='C1', label='Floating Payment'),
                                mpatches.Patch(color='C2', label='Fixing Date')], loc='best')
    
    # Add secondary x axis to display time in years
    # Obtain dimensions of primary x axis
    divider = make_axes_locatable(ax)
    ax_cb = divider.new_vertical(size="5%", pad=0.4, pack_start=True)
    
    # Configure secondary x axis
    ax2 = fig.add_axes(ax_cb)
    ax2.set_xlim(0, tenor_structure[-1] + 1/10)
    ax2.yaxis.set_visible(False) # hide the yaxis
    ax2.set_xlabel('Time $t$ (years)')
    xticks2_locs = range(int(tenor_structure[-1])+1)
    xticks2_labels = range(int(tenor_structure[-1])+1)
    ax2.set_xticks(xticks2_locs, xticks2_labels)
    ax2.grid(True, which='major', color='gray')
    ax2.axhline(y = 0, color='gray')
    
    # Display and then close figure
    plt.show()
    plt.close()
    
def plot_Bermudan_swaption_timeline(fixed_rate: float,
                                    floating_rates: list,
                                    swap_type: str,
                                    tenor_structure: list
                                   ) -> None:
    """
    Info: This function plots the timeline of a Bermudan forward-start interest 
        rate swap option in order to illustrate the tenor.
    
    Input:
        fixed_rate: the fixed interest rate of the forward-start swap.
        
        swap_type: whether the swap is a payer or receiver forward-start swap.
        
        tenor_structure: a list containing the underlying forward-start swap 
            starting date as the first entry and the payment dates as the 
            remaining entries.
            
    Output:
        A plot of the Bermudan swaption timeline.
    """
    
    # Create figure
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_xlim(0, tenor_structure[-1]*(1.1))
    
    
    # Adjust y axis limits and size of arrows
    if fixed_rate > np.max(floating_rates):
        ax.set_ylim(-1.1*fixed_rate, 1.1*fixed_rate)
    else:
        ax.set_ylim(-1.1*np.max(floating_rates), 1.1*np.max(floating_rates))
    
    # Configure main x axis
    xticks_locs = tenor_structure
    xticks_labels = np.concatenate((r'$T_0$', 
                                    np.array([r'$T_0$' + '$_+$' + f'$_{idx}$' 
                                              for idx in range(1,len(tenor_structure)-1)], 
                                             dtype=str)), axis=None)
    xticks_labels = np.concatenate((xticks_labels, r'$T_M$'), axis=None)
    ax.set_xticks(xticks_locs, xticks_labels)
    ax.grid(True, which='both')
    ax.set_title(f'Bermudan {swap_type.capitalize()} Forward-Start Swaption Timeline')
    ax.set_ylabel('Interest Rate')
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.axhline(y=0, color='gray', lw=2)
    fig.autofmt_xdate()#rotation=45, ha='center')
    
    # Plot arrows at payment dates 
    if swap_type.lower() == 'payer':
        for count, T_payment in enumerate(tenor_structure[1:]):
            ax.add_patch(curly_arrow(ax, (T_payment, 0), (T_payment, floating_rates[count]), 'C0', arrow_type='curly'))
            ax.add_patch(curly_arrow(ax, (T_payment, 0), (T_payment, -fixed_rate), 'C1', arrow_type='straight'))
            ax.scatter(tenor_structure[:-1], np.zeros(len(tenor_structure)-1), color='C2', marker='o', s=225)
            ax.scatter(tenor_structure[:-1], np.zeros(len(tenor_structure)-1), color='C3', marker='x', s=75)
            plt.legend(handles=[mpatches.Patch(color='C0', label='Floating Payment'),
                                mpatches.Patch(color='C1', label='Fixed Payment'),
                                mpatches.Patch(color='C2', label='Fixing Date'),
                                mpatches.Patch(color='C3', label='Monitor Date')], loc='best')
    elif swap_type.lower() == 'receiver':
        for count, T_payment in enumerate(tenor_structure[1:]):
            ax.add_patch(curly_arrow(ax, (T_payment, 0), (T_payment, fixed_rate), 'C0', arrow_type='straight'))
            ax.add_patch(curly_arrow(ax, (T_payment, 0), (T_payment, -floating_rates[count]), 'C1', arrow_type='curly'))
            ax.scatter(tenor_structure[:-1], np.zeros(len(tenor_structure)-1), color='C2', marker='o', s=225)
            ax.scatter(tenor_structure[:-1], np.zeros(len(tenor_structure)-1), color='C3', marker='x', s=75)
            plt.legend(handles=[mpatches.Patch(color='C0', label='Fixed Payment'), 
                                mpatches.Patch(color='C1', label='Floating Payment'),
                                mpatches.Patch(color='C2', label='Fixing Date'),
                                mpatches.Patch(color='C3', label='Monitor Date')], loc='best')
    
    # Add secondary x axis to display time in years
    # Obtain dimensions of primary x axis
    divider = make_axes_locatable(ax)
    ax_cb = divider.new_vertical(size="5%", pad=0.4, pack_start=True)

    # Configure secondary x axis
    ax2 = fig.add_axes(ax_cb)
    ax2.set_xlim(0, tenor_structure[-1]*(1.1))
    ax2.yaxis.set_visible(False) # hide the yaxis
    ax2.set_xlabel('Time $t$ (years)')
    xticks2_locs = range(int(tenor_structure[-1])+1)
    xticks2_labels = range(int(tenor_structure[-1])+1)
    ax2.set_xticks(xticks2_locs, xticks2_labels)
    ax2.xaxis.set_minor_locator(MultipleLocator(1/12))
    ax2.grid(True, which='major', color='gray')
    ax2.grid(True, which='minor')
    ax2.axhline(y = 0, color='gray')
    
    # Display and then close figure
    plt.show()
    plt.close()

def visualize_neural_network(experiment_dir: str,
                             file_name: str,
                             neural_network: object
                            ):
    figures_dir = os.path.join(experiment_dir, 'Figures\\')
    
    visualizer(neural_network, filename=figures_dir+f'\\{file_name}', 
               format='png', view=True)
    
