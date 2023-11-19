# %% import files
import scipy.io as sio
from tkinter import filedialog, Tk
import cv2 as cv
import numpy as np
from IPython.display import Image, display
import os
import pandas as pd
import matlab.engine
import pickle
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
# %%

class ElectricalDataPr():
    def __init__(self, pickled_file_path):
        """initiaes class with pickle file from matlab data (already processed by load_tools.py)"""
        self.directory = os.path.dirname(pickled_file_path)
        self.fname = os.path.basename(pickled_file_path).split('/')[-1]
        with open(pickled_file_path, 'rb') as file:
            self.metadata, self.trial_data = pickle.load(file)
    def this_trial(self,i,j):
        """
        Returns  the trial data for the (i,j)th data
        """
        
        nom_voltage = self.metadata['V_in'][i,j,:]
        nom_voltage = np.unique(nom_voltage)
        delay = self.metadata['delay'][i,j,:]
        delay = np.unique(delay)
        light_level = self.metadata['light_level'][i,j,:]
        light_level = np.unique(light_level)

        if (len(nom_voltage), len(delay), len(light_level))!=(1,1,1):
            print("not a trial", nom_voltage, delay, light_level)
        else:
            trial = TrialPr(self.trial_data[i,j,:], nom_voltage=nom_voltage[0], delay = delay[0], light_level=light_level[0])
            return trial


# %% 
class TrialPr():
    def __init__(self, this_trial, nom_voltage, delay, light_level, 
                 initial_num =1, set_size = 5 ):
        """
        this_trial: data for a given parameter config
        nom_voltage: voltage supplied (nominal)
        delay: in ms, delay value supplied to arduino
        light_level: light_level supplied (not very relevant)
        initial_num: how many recordings in first run
        set_size: this is the number after which recording parameters changed. 5 by default
        """
        self.data = this_trial
        self.nom_voltage = nom_voltage
        self.delay = delay
        self.light_level = light_level
        self.initial_num = initial_num
        self.set_size = set_size
    def separate_data(self):
        """This separates data into different kinds of recordings"""
        num_recordings = self.data.shape[0]
        self.initial_recording = self.data[0]
        num_sets = (num_recordings-self.initial_num)//self.set_size
        self.hr_data = {'charge_A':[], 'discharge_A':[], 'charge_B':[], 'discharge_B':[]}
        keys = list(self.hr_data)
        for set_idx in range(num_sets):
            self.hr_data[keys[set_idx]] = self.data[self.initial_num+set_idx*self.set_size:self.initial_num+(set_idx+1)*self.set_size ]
    def align_to_trigger(self, data:'np.ndarray[pd.DataFrame]', trigger_name = 'trigger', rising_edge = False, 
                         thresh = 2.5):
        shifts = np.zeros(len(data))
        for ind, this_trial in enumerate(data):
            trigger_signal = this_trial[trigger_name]
            high_indxs = trigger_signal>=thresh
            low_indxs = trigger_signal< thresh
            high_indxs = high_indxs.to_numpy()
            low_indxs = low_indxs.to_numpy()
            if rising_edge:
                trigger_pts = high_indxs[1:] & low_indxs[:-1]
            else:
                trigger_pts = high_indxs[1:] & low_indxs[:-1]
            trigger_pts = np.flatnonzero(trigger_pts)
            zero_time_indx = np.abs(this_trial['time']).argmin()
            if len(trigger_pts)>=1:
                chosen_triggerpt = trigger_pts[np.abs(trigger_pts-zero_time_indx).argmin()]
                new_time = this_trial['time'] - this_trial['time'][chosen_triggerpt]
                this_trial['time_og'] = this_trial['time']
                shifts[ind] = this_trial['time'][chosen_triggerpt]
                this_trial['time'] = new_time
        return data, shifts
    def plot_time(self, data:'np.ndarray[pd.DataFrame]', time_window = None):
        num_of_trials = len(data)
        fig, ax = plt.subplots(3,1, sharex=True)
        cmap = mpl.cm.get_cmap('viridis')
        colors = cmap(np.linspace(0,1,num_of_trials))
        for ind, trial_data in enumerate(data):
            if time_window is None:
                time_window = np.array([ trial_data['time'][0], trial_data['time'].iloc[-1]])
            trial_data = trial_data[ (trial_data['time']>=time_window[0])& (trial_data['time']<=time_window[1])]
            ax[0].plot(trial_data['time']*1000, trial_data['V_dev'], color = colors[ind])
            ax[1].plot(trial_data['time']*1000, trial_data['trigger'])
            ax[2].plot(trial_data['time']*1000, trial_data['V_rout'])
        ax[2].set_xlabel('Time (ms)')
        ax[0].set_ylabel('Device voltage(V)')
        ax[1].set_ylabel('Trigger signal (V)')
        ax[2].set_ylabel('V_rout (V)')
    def align_all_to_trigger(self):
        channel = ['trigger','trigger','trigger_B','trigger_B']
        rising_edge = [False, True, False, True]
        classification = ['charge_A', 'discharge_A', 'charge_B', 'discharge_B']
        for i, tag in enumerate(classification):
            this_set = self.hr_data[tag]
            if len(this_set)>0:
                self.align_to_trigger(this_set, channel[i], rising_edge[i])


# %%
root = Tk()
root.withdraw()

pickle_file = filedialog.askopenfilenames(filetypes=[("Pickle py files", "*.py")])[0]


# %%
e_pr = ElectricalDataPr(pickle_file)
# %%
e_pr.trial_data.shape

# %%
this_trial = e_pr.this_trial(0,0)
# %%
this_trial.separate_data()
# %%
this_trial.plot_time(this_trial.hr_data['charge_A'])
# %%
this_trial.plot_time(this_trial.hr_data['charge_A'], [-0.00001, 0.00005])
# %%
# this_trial.align_to_trigger(this_trial.hr_data['charge_A'], rising_edge=True)
# 
this_trial.align_all_to_trigger()

# %%
