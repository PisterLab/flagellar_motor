# The point of this script is to read csv data with angle and time and compute different values, plot them, and output csv for desired plots
# %% 
import cv2 as cv
import numpy as np
import tkinter as tk
from tkinter import filedialog, Tk
from math import sqrt, atan2, degrees
import csv
import matplotlib.pyplot as plt
plt.style.use("ggplot")
# from IPython.display import Image, display
import os
import pandas as pd
import sys
import matplotlib as mpl
# %% 

design_module_path = 'c:\\Users\\mbustamante\\Box Sync\\Research\\Flagellar Motor\\code\\rotary_inchworm'
if design_module_path not in sys.path:
    sys.path.append(design_module_path) 
import masks._20230328_mbustama_kr.krypton_analysis as kr_ana
# # %%
# root = Tk()
# root.withdraw()

# csv_file = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv")])[0]




# %% 
def get_design_parameter(dev, parameter):
    label =  dev.split('_')[-1]
    lbl_ind = kr_ana.labels.index(label)
    return kr_ana.draw_args[parameter][lbl_ind]

# %% test parameters for different functions





# %% 
class AngularDataPr():
    """
    Class to process csv files with angular data (and down the line linear data)

    Attributes
    ----------
    csv_file: file_path to csv file containing motion data
    directory: directory path were csv file (and usually mp4 files) are located
    fname: just the base file name without directory data
    data: pandas dataframe with csvdata
    time_cols: column headers starting with time
    device_name: obtained from fname, contains chip ID and device ID
    """

    def __init__(self, csv_file):
        """
        Parameters
        ----------
        csv_file: file path to csv file containing columns for time, angle, and (sometimes) x and y motion

        """
        self.csv_file = csv_file
        self.directory = os.path.dirname(csv_file)
        self.fname = os.path.basename(csv_file).split('/')[-1]
        self.data = pd.read_csv(csv_file)
        self.time_cols = [col for col in self.data if col.startswith('Time')]
        self.device_name = self.fname.split('.')[0]
        self.figs = []
    def get_time_angle(self, i):
        """ Returns the time and angle data for index i"""
        time = self.data[self.time_cols[i]]
        angle = self.data.iloc[:,self.data.columns.get_loc(self.time_cols[i])+1]
        return time, angle
    
    def process_all(self, time_correction_factor = 1, visualize = False):
        self.single_processors = []
        for i, time_col in enumerate(self.time_cols):
            time,angle = self.get_time_angle(i)
            time = time_correction_factor*time
            angular_sc = AngularDataSC(time, angle)
            angular_sc.compute_all()
            angular_sc.metadata_from_column_name(time_col)
            self.single_processors.append(angular_sc)
            if visualize:
                angular_sc.visualize()

    def save_FOMS(self):
        self.summary_data = pd.DataFrame(columns = [
            'delay_time', 
            'nom_voltage',
            'average_speed', 
            'reg_speed',
            'avg_step_size',
            'median_step_size',
            'std_step_size',
            'avg_step_period',
            'median_step_period',
            'std_step_period'])
        for i in range(len(self.single_processors)):
            angular_sc = self.single_processors[i] #type: AngularDataSC
            delay = angular_sc.delay
            nom_voltage = angular_sc.nominal_voltage
            average_speed = angular_sc.average_speed
            reg_speed = angular_sc.speed_slope
            step_sizes = angular_sc.step_sizes
            avg_step_size = np.mean(step_sizes)
            median_step_size = np.median(step_sizes)
            std_step_size = np.std(step_sizes)
            step_periods = angular_sc.step_periods
            avg_step_period = np.mean(step_periods)
            median_step_period = np.median(step_periods)
            std_step_period = np.std(step_periods)
            this_dict = {
            'delay_time': delay, 
            'nom_voltage': nom_voltage,
            'average_speed': average_speed, 
            'reg_speed': reg_speed,
            'avg_step_size': avg_step_size,
            'median_step_size': median_step_size,
            'std_step_size': std_step_size,
            'avg_step_period': avg_step_period,
            'median_step_period': median_step_period,
            'std_step_period': std_step_period
            }           
            
            self.summary_data.loc[i] = list(this_dict.values()) 
            print(i)
        self.compute_frequency()
    def compute_frequency(self):
        """Adds a column for the frequency to the data structure """
        delay_time = self.summary_data['delay_time']
        frequency = 1/(4*delay_time*1e-3)
        self.summary_data['frequency'] = frequency
    def plot_FOMS(
            self,y_var, primary_var = 'nom_voltage', secondary_var = 'delay_time', 
            xlabel = None, ylabel =None, legend_label = None,
            colormap = mpl.cm.get_cmap('viridis')
            ):
        """ plots figure of merits, with primary var as x axis, secondary var as legend"""
        if not hasattr(self, 'summary_data'):
            raise Exception('Call save_FOMS() before this function')
            
        secondary_set = self.summary_data[secondary_var].unique()
        legend_colors = colormap(np.linspace(0,1,len(secondary_set)))
        fig, ax = plt.subplots()
        arts = []
        for i, sec_val in enumerate(secondary_set):
            subdata = self.summary_data[self.summary_data[secondary_var] == sec_val]
            xvals = subdata[primary_var]
            yvals = subdata[y_var]
            color = legend_colors[i, :-1].reshape(1,3)
            pts = ax.scatter(xvals, yvals, color = color, label = sec_val )
            arts.append(pts)

        if xlabel is None:
            xlabel = primary_var
        if ylabel is None:
            ylabel = y_var
        if legend_label is None:
            legend_label = secondary_var

        labels = [l.get_label() for l in arts]

        # clset = set(zip(arts, labels))
        # handles = [plt.plot([], color = c, ls = "", marker = 'o' )[0] for c, l in clset]
        # labels = [l for c,l in clset]
        
        # ax.legend()
        ax.legend(title = legend_label)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        self.figs.append(fig)




    # def find_step_indexes(self, time _ind:int,  thresh= 0.1):
    #     """
    #     Computes and returns locations for steps
    #     Parameters
    #     ----------
    #     time_ind: index for which time column to do this for 
    #     thresh: threshold to consider an amount of motion a step 
    #     """
    #     time = self.data[self.time_cols[time_ind]]
    #     angle = self.data.iloc[:,self.data.columns.get_loc(self.time_cols[time_ind])+1]
    #     steps = np.diff(angle)
    #     indxs = np.where(np.abs(steps)>= thresh)[0]
        
    #     # else: 
    #     #     fig = None
    #     return indxs, time[indxs]
    # def find_step_sizes(self, time_ind, thresh = 0.1):


    # def get_step_size(self):
    #     """ computes step sizes"""
    #     angled_steps = angle[step_indxs]
    #     step_sizes = np.diff(angled_steps)
    #     fig = None
    #     fig2 = None
    #     if plot:
    #         fig2, ax = plt.subplots()
    #         ax.hist(np.diff(angle[step_indxs]))
    #         ax.set_xlabel('Step size(degrees)')
    #     return step_sizes, [fig, fig2]


# def differentiate(time, angle, plot = False):
#     """discrete derivative for step sizes0"""
#     dangle = np.diff(angle)
#     if plot:
#         plt.plot(time[0:-1], dangle)
#     return dangle

class AngularDataSC():
    """ Class to include the methods for every series of time vs angle (can include x and y in future)"""
    def __init__(self, time, angle, step_thresh = 0.1):
        """
        Parameters
        ----------
        time: time series corresponding to angle 
        angle: cummulative angle steps
        step_thresh: threshold for steps later
        """
        self.time = np.array(time)
        self.angle = np.array(angle)
        self.step_thresh = step_thresh 
        self.figs = {}
    def compute_all(self):
        self.find_step_indexes()
        self.find_step_sizes()
        self.linear_regression()
        self.get_average_speed()
    def find_step_indexes(self):
        steps = np.diff(self.angle)
        self.step_indxs = np.where(np.abs(steps)>= self.step_thresh)[0]
        self.step_periods = np.diff(self.time[self.step_indxs])
        # if plot:
        #     fig, ax = plt.subplots()
        #     ax.plot(self.time, self.angle)
        #     plt.vlines(x = time[self.step_indxs], ymin = 0, ymax = np.max(self.angle), colors = 'gray')
        #     fig2, ax = plt.subplots()
        #     ax.hist(np.diff(time[self.step_indxs]))
        #     ax.set_xlabel('Step time (s)')
        #     self.figs['Step time'] = fig
        #     self.figs['']
        return self.step_indxs
    def find_step_sizes(self):
        """Must be called after finding step_indxs"""
        angled_steps = self.angle[self.step_indxs]
        self.step_sizes = np.diff(angled_steps)
        return self.step_sizes
    def linear_regression(self, force_intcpt = False):
        """ Computes the speed using linear regression
        Parameters
        ----------
        force_intcp: if True sets intercept at 0
        """
        time = self.time
        angle = self.angle
        if not force_intcpt:
            A = np.vstack([time, np.ones(len(time))]).T
            slope, intercept = np.linalg.lstsq(A, angle, rcond = None)[0]
        else:
            A = time[:, np.newaxis]
            slope = np.linalg.lstsq(A, angle, rcond = None)[0][0]
            intercept = 0
        self.speed_slope = slope
        self.lin_reg_intcpt = intercept
        return slope, intercept 
    def get_average_speed(self):
        """
        Computes average speed simply delta angle/ delta time
        """
        self.average_speed = (self.angle[-1]- self.angle[0])/(self.time[-1] - self.time[0])
        return self.average_speed
    def load_metadata(self, nominal_voltage, delay, shuttle_radius = None, expected_step_size = None ):
        self.nominal_voltage = nominal_voltage
        self.delay = delay
        if shuttle_radius:
            self.shuttle_radius = shuttle_radius
        if expected_step_size:
            self.expected_step = expected_step_size
    def metadata_from_column_name(self, column_name:str):
        self.nominal_voltage = int([v for v in column_name.split(' ') if 'V' in v][0].split('V')[0])
        try:
            delay = float([v for v in column_name.split(' ') if 'ms' in v][0].split('ms')[0])
        except ValueError:
            delay = float([v for v in column_name.split(' ') if 'Ms' in v][0].split('Ms')[0])
        self.delay = delay


    def visualize(self):
        """ Produces plots to visualize computation"""
        # show time vs angle with speed and steps 
        fig, ax = plt.subplots(figsize=(5,5))
        fig2, ax2 = plt.subplots(2,2, figsize = (10,8))
        ax.plot(self.time, self.angle)
        if hasattr(self, 'step_indxs'):
            ax.vlines(x = self.time[self.step_indxs], ymin = 0, 
                      ymax = np.max(self.angle), colors = 'gray')
            period_bins, period_freqs, _ = ax2[0,0].hist(self.step_periods)
        if hasattr(self, 'step_sizes'):
            ax2[0,1].hist(self.step_sizes)
            ax2[1,1].scatter(self.step_periods, self.step_sizes)
        if hasattr(self, 'speed_slope'):
            ax.plot(self.time, self.time*self.speed_slope + self.lin_reg_intcpt, '-')
        if hasattr(self, 'nominal_voltage'):
            fig.suptitle(f'Nominal {self.nominal_voltage}: V with delay = {self.delay}')
            ax2[0,0].vlines(x=[2*self.delay *1e-3], ymin = 0, ymax =period_freqs.max(), color = 'black' )
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (degrees)')
        ax2[0,0].set_xlabel('Step period (s)')
        ax2[0,1].set_xlabel('Step size (degrees)')
        ax2[1,1].set_xlabel('Step period (s)')    
        ax2[1,1].set_ylabel('Step size (degrees)')
        self.figs = {'Motion': fig, 'Step distributions': fig2}


# %% 

# processor = AngularDataPr(csv_file)
# time_correction_factor =30/41.25
# processor.process_all(time_correction_factor=time_correction_factor, visualize = False)
# processor.save_FOMS()

# # %% 
# processor.plot_FOMS(y_var='average_speed', primary_var='nom_voltage', secondary_var='delay_time')
# processor.plot_FOMS(y_var='reg_speed', primary_var='delay_time', secondary_var='nom_voltage')

# # %%

# processor.plot_FOMS(y_var='average_speed', primary_var='nom_voltage', secondary_var='frequency')
# processor.plot_FOMS(y_var='reg_speed', primary_var='frequency', secondary_var='nom_voltage')
# %%
# for i in range(0,len(processor.time_cols)):
#     time_correction_factor = 30/41.25
#     time,angle = processor.get_time_angle(i)
#     time = time_correction_factor*time
#     subprocessor = AngularDataSC(time, angle, 0.1)

#     subprocessor.compute_all()
#     subprocessor.metadata_from_column_name(processor.time_cols[i])
#     subprocessor.visualize()

# def linear_Regression(time,angle, force_intcpt = False, plot =False):
#     """ computes speed based on linear regression of value"""
#     if not force_intcpt:
#         A = np.vstack([time, np.ones(len(time))]).T
#         slope, intercept = np.linalg.lstsq(A, angle, rcond = None)[0]
#     else:
#         A = time[:, np.newaxis]
#         slope = np.linalg.lstsq(A, angle, rcond = None)[0][0]
#         intercept = 0
#     if plot:
#         fig, ax = plt.subplots()
#         ax.plot(time, angle)
#         ax.plot(time, time*slope + intercept)
#         ax.set_xlabel('Time (s)')
#         ax.set_ylabel('Angle (degrees)')

#     return slope, intercept, fig
# %%
# time_diff_list = []
# step_size_list = []
# fig, ax = plt.subplots()
# hist_bins = 1/41.25*np.arange(0, 50, 2)
# slopes = []
# intercepts = []
# for i in range(0,len(time_cols)):
#     time_correction_factor = 30/41.25
#     indx, time, figs = find_step_indexes(df[time_cols[i]]*time_correction_factor, df.iloc[:,df.columns.get_loc(time_cols[i])+1], plot = True, thresh = 0.2)
#     step_sizes, _ = get_step_size(indx,df[time_cols[i]]*time_correction_factor, df.iloc[:,df.columns.get_loc(time_cols[i])+1], plot = True )
#     step_size_list.append(step_sizes)
#     slope, intercept, _ = linear_Regression(df[time_cols[i]]*time_correction_factor, df.iloc[:,df.columns.get_loc(time_cols[i])+1], plot = True, force_intcpt = False)
#     slopes.append(slope)
#     time_diffs = np.diff(time)
#     fig2, ax2 = plt.subplots()
#     ax2.scatter(time_diffs, step_sizes)
#     filtered_time_diffs = time_diffs[np.logical_and(time_diffs<0.5,time_diffs>1/41.25)]
#     time_diff_list.append(time_diffs)
#     ax.hist(filtered_time_diffs, bins = hist_bins,  density= True, rwidth =0.8)
# %% get medians for step sizes
# step_sizes_meds= []
# nominal_Vs = []
# nominal_delays = []
# corresponding_step_times = []
# for i in range(0,len(time_cols)):
#     nominal_V = int([v for v in time_cols[i].split(' ') if 'V' in v][0].split('V')[0])
#     nominal_Vs.append(nominal_V)
#     try:
#         delay = float([v for v in time_cols[i].split(' ') if 'ms' in v][0].split('ms')[0])
#     except ValueError:
#         delay = float([v for v in time_cols[i].split(' ') if 'Ms' in v][0].split('Ms')[0])
#     nominal_delays.append(delay)
#     sizes = step_size_list[i]
#     med = np.median(sizes)
#     step_sizes_meds.append(med)
# # %% step size vs voltage
# rad = get_design_parameter(dev, 'shuttle_ro')
# fig, ax = plt.subplots()
# ax.scatter(nominal_Vs, step_sizes_meds)
# ax.set_xlabel('Input Voltage (V) (nominal)')
# ax.set_ylabel('Median step size (degrees)')
# secax = ax.secondary_yaxis('right', functions= (lambda x: x*np.pi/180 *rad, lambda y: y / (np.pi/180 *rad)))
# secax.set_ylabel('Median step size (um)')

# # %% moving slope vs voltage
# fig, ax = plt.subplots()
# ax.scatter(nominal_Vs, slopes)
# ax.set_xlabel('Input Voltage (V) (nominal)')
# ax.set_ylabel('Angular velocity(degrees/s)')
# secax = ax.secondary_yaxis('right', functions= (lambda x: x*np.pi/180 *rad, lambda y: y / (np.pi/180 *rad)))
# secax.set_ylabel('Linear speed (um/s)')

# # %% moving speed vs delay
# fig, ax = plt.subplots()
# nominal_delays = np.array(nominal_delays)

# slopes = np.array(slopes)

# unique_Vs = set(nominal_Vs)
# for volts in unique_Vs:
#     filtered_inds = [int(i) for i in range(len(nominal_Vs)) if nominal_Vs[i] == volts]
#     ax.scatter(nominal_delays[filtered_inds], slopes[filtered_inds])
# ax.set_xlabel('Delay time (s)')
# ax.set_ylabel('Angular velocity(degrees/s)')
# secax = ax.secondary_yaxis('right', functions= (lambda x: x*np.pi/180 *rad, lambda y: y / (np.pi/180 *rad)))
# secax.set_ylabel('Linear speed (um/s)')

# # %% 

# # %% moving speed vs delay
# fig, ax = plt.subplots()
# nominal_delays = np.array(nominal_delays)

# slopes = np.array(slopes)

# unique_Vs = set(nominal_Vs)
# for volts in unique_Vs:
#     filtered_inds = [int(i) for i in range(len(nominal_Vs)) if nominal_Vs[i] == volts]
#     ax.scatter(1000*0.5/nominal_delays[filtered_inds], slopes[filtered_inds])
# ax.set_xlabel('Step Frequency (Hz)')
# ax.set_ylabel('Angular velocity(degrees/s)')
# secax = ax.secondary_yaxis('right', functions= (lambda x: x*np.pi/180 *rad, lambda y: y / (np.pi/180 *rad)))
# secax.set_ylabel('Linear speed (um/s)')
# # %%
# step_times = np.concatenate(time_diff_list).flatten()
# step_times = step_times[step_times<0.25]
# plt.hist(step_times)
# %% 


# %% Export as 
# import tikzplotlib
# tikzplotlib.save(os.path.join(directory,'speed_v_voltage.tex'), figure =  fig)
# %%

