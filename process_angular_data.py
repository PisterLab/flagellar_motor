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
# %% 

design_module_path = 'c:\\Users\\mbustamante\\Box Sync\\Research\\Flagellar Motor\\code\\rotary_inchworm'
if design_module_path not in sys.path:
    sys.path.append(design_module_path) 
import masks._20230328_mbustama_kr.krypton_analysis as kr_ana
# %%
root = Tk()
root.withdraw()

csv_file = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv")])[0]

# %% 
directory = os.path.dirname(csv_file)
fname = os.path.basename(csv_file).split('/')[-1]
# %%
df = pd.read_csv(csv_file)
# %%
time_cols = [col for col in df if col.startswith('Time')]
# %%

# for time_col_name in time_cols[0:1]:
#     time = df[time_col_name]
#     angle = df.iloc[df.columns.get_indexer(time_col_name)]
# %%




# %% 
def get_design_parameter(dev, parameter):
    label =  dev.split('_')[-1]
    lbl_ind = kr_ana.labels.index(label)
    return kr_ana.draw_args[parameter][lbl_ind]

# %% test parameters for different functions

# %%
dev = fname.split('.')[0]


# %%
def get_step_size(step_indxs, time, angle, plot = False):
    """ computes step size"""
    angled_steps = angle[step_indxs]
    step_sizes = np.diff(angled_steps)
    fig = None
    fig2 = None
    if plot:
        fig2, ax = plt.subplots()
        ax.hist(np.diff(angle[step_indxs]))
        ax.set_xlabel('Step size(degrees)')
    return step_sizes, [fig, fig2]


def differentiate(time, angle, plot = False):
    """discrete derivative for step sizes0"""
    dangle = np.diff(angle)
    if plot:
        plt.plot(time[0:-1], dangle)
    return dangle

def find_step_indexes(time, angle, thresh = 0.1, plot = False):
    """returns the locations in which steps occur"""
    steps = differentiate(time, angle, False)
    indxs = np.where(np.abs(steps)>= thresh)[0]
    fig = None
    fig2 = None
    if plot:
        fig, ax = plt.subplots()
        ax.plot(time, angle)
        plt.vlines(x = time[indxs], ymin = 0, ymax = np.max(angle), colors = 'gray')
        fig2, ax = plt.subplots()
        ax.hist(np.diff(time[indxs]))
        ax.set_xlabel('Step time (s)')
    else: 
        fig = None
    return indxs, time[indxs], [fig, fig2]

def linear_Regression(time,angle, force_intcpt = False, plot =False):
    """ computes speed based on linear regression of value"""
    if not force_intcpt:
        A = np.vstack([time, np.ones(len(time))]).T
        slope, intercept = np.linalg.lstsq(A, angle, rcond = None)[0]
    else:
        A = time[:, np.newaxis]
        slope = np.linalg.lstsq(A, angle, rcond = None)[0][0]
        intercept = 0
    if plot:
        fig, ax = plt.subplots()
        ax.plot(time, angle)
        ax.plot(time, time*slope + intercept)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (degrees)')

    return slope, intercept, fig

# %%
time_diff_list = []
step_size_list = []
fig, ax = plt.subplots()
hist_bins = 1/41.25*np.arange(0, 50, 2)
slopes = []
intercepts = []
for i in range(0,len(time_cols)):
    time_correction_factor = 30/41.25
    indx, time, figs = find_step_indexes(df[time_cols[i]]*time_correction_factor, df.iloc[:,df.columns.get_loc(time_cols[i])+1], plot = True, thresh = 0.2)
    step_sizes, _ = get_step_size(indx,df[time_cols[i]]*time_correction_factor, df.iloc[:,df.columns.get_loc(time_cols[i])+1], plot = True )
    step_size_list.append(step_sizes)
    slope, intercept, _ = linear_Regression(df[time_cols[i]]*time_correction_factor, df.iloc[:,df.columns.get_loc(time_cols[i])+1], plot = True, force_intcpt = False)
    slopes.append(slope)
    time_diffs = np.diff(time)
    fig2, ax2 = plt.subplots()
    ax2.scatter(time_diffs, step_sizes)
    filtered_time_diffs = time_diffs[np.logical_and(time_diffs<0.5,time_diffs>1/41.25)]
    time_diff_list.append(time_diffs)
    ax.hist(filtered_time_diffs, bins = hist_bins,  density= True, rwidth =0.8)
# %% get medians for step sizes
step_sizes_meds= []
nominal_Vs = []
nominal_delays = []
corresponding_step_times = []
for i in range(0,len(time_cols)):
    nominal_V = int([v for v in time_cols[i].split(' ') if 'V' in v][0].split('V')[0])
    nominal_Vs.append(nominal_V)
    try:
        delay = float([v for v in time_cols[i].split(' ') if 'ms' in v][0].split('ms')[0])
    except ValueError:
        delay = float([v for v in time_cols[i].split(' ') if 'Ms' in v][0].split('Ms')[0])
    nominal_delays.append(delay)
    sizes = step_size_list[i]
    med = np.median(sizes)
    step_sizes_meds.append(med)
# %% step size vs voltage
rad = get_design_parameter(dev, 'shuttle_ro')
fig, ax = plt.subplots()
ax.scatter(nominal_Vs, step_sizes_meds)
ax.set_xlabel('Input Voltage (V) (nominal)')
ax.set_ylabel('Median step size (degrees)')
secax = ax.secondary_yaxis('right', functions= (lambda x: x*np.pi/180 *rad, lambda y: y / (np.pi/180 *rad)))
secax.set_ylabel('Median step size (um)')

# %% moving slope vs voltage
fig, ax = plt.subplots()
ax.scatter(nominal_Vs, slopes)
ax.set_xlabel('Input Voltage (V) (nominal)')
ax.set_ylabel('Angular velocity(degrees/s)')
secax = ax.secondary_yaxis('right', functions= (lambda x: x*np.pi/180 *rad, lambda y: y / (np.pi/180 *rad)))
secax.set_ylabel('Linear speed (um/s)')

# %% moving speed vs delay
fig, ax = plt.subplots()
nominal_delays = np.array(nominal_delays)

slopes = np.array(slopes)

unique_Vs = set(nominal_Vs)
for volts in unique_Vs:
    filtered_inds = [int(i) for i in range(len(nominal_Vs)) if nominal_Vs[i] == volts]
    ax.scatter(nominal_delays[filtered_inds], slopes[filtered_inds])
ax.set_xlabel('Delay time (s)')
ax.set_ylabel('Angular velocity(degrees/s)')
secax = ax.secondary_yaxis('right', functions= (lambda x: x*np.pi/180 *rad, lambda y: y / (np.pi/180 *rad)))
secax.set_ylabel('Linear speed (um/s)')

# %% 

# %% moving speed vs delay
fig, ax = plt.subplots()
nominal_delays = np.array(nominal_delays)

slopes = np.array(slopes)

unique_Vs = set(nominal_Vs)
for volts in unique_Vs:
    filtered_inds = [int(i) for i in range(len(nominal_Vs)) if nominal_Vs[i] == volts]
    ax.scatter(1000*0.5/nominal_delays[filtered_inds], slopes[filtered_inds])
ax.set_xlabel('Step Frequency (Hz)')
ax.set_ylabel('Angular velocity(degrees/s)')
secax = ax.secondary_yaxis('right', functions= (lambda x: x*np.pi/180 *rad, lambda y: y / (np.pi/180 *rad)))
secax.set_ylabel('Linear speed (um/s)')
# %%
step_times = np.concatenate(time_diff_list).flatten()
step_times = step_times[step_times<0.25]
plt.hist(step_times)
# %% 


# %% Export as 
import tikzplotlib
tikzplotlib.save(os.path.join(directory,'speed_v_voltage.tex'), figure =  fig)
# # %%

# %%


# %%
