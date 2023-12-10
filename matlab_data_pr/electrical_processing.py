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
        self.trials = np.empty(self.trial_data.shape[0:2], dtype = TrialPr)
    def this_trial(self,i,j):
        """
        Creates the trial data for the (i,j)th data
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

    def analyze_all_trials(self):
        """Runs through all the trials in a file"""
        for i in range(self.trial_data.shape[0]):
            for j in range(self.trial_data.shape[1]):
                this_trial = self.this_trial(i,j)
                this_trial.analyze()
                self.trials[i,j] = this_trial

    def plot_voltage_nom(self, cat = 'charge_A', errors = True):
        """plots nominal vs input voltage over all trials"""
        cmap = mpl.cm.get_cmap('viridis')
        colors = cmap(np.linspace(0,1,self.trials.shape[1]))
        markers = [',', '+', '.', 'o', '*']
        nom_vs = np.empty(self.trials.shape)
        measured_voltages = np.empty((*self.trials.shape,5))
        measured_stds = np.empty((*self.trials.shape,5))
        fig, ax = plt.subplots()
        ax.set_xlabel('Nominal input voltage (V)')
        ax.set_ylabel('Measured input voltage')
        for i in range(self.trials.shape[0]):
            for j in range(self.trial_data.shape[1]):
                this_trial = self.trials[i,j]
                vs = [meas[0] for meas in this_trial.V_in_meas[cat]]
                vs = np.array(vs)
                measured_voltages[i,j,:] = vs
                stds = [meas[1] for meas in this_trial.V_in_meas[cat]]
                stds = np.array(stds)
                measured_voltages[i,j,:] = stds
                if errors:
                    ax.errorbar(this_trial.nom_voltage * np.ones(stds.shape), vs, yerr = stds, color = colors[j], marker = markers[i%5],fmt = 'o')
                else:
                    ax.scatter(this_trial.nom_voltage * np.ones(stds.shape), vs, color = colors[j], marker = markers[i%5])
                nom_vs[i,j] = this_trial.nom_voltage
        ax.plot(nom_vs[i,:], nom_vs[i,:],'--', color = 'black')
    def plot_vin_vs_tau(self, cat = 'charge_A', tau_type = 'current', plot_det = False):
        """ plots taus for given category and current against input voltage"""
        cmap = mpl.cm.get_cmap('viridis')
        colors = cmap(np.linspace(0,1,self.trials.shape[1]))
        markers = [',', '+', '.', 'o', '*']
        fig, ax = plt.subplots()
        ax.set_xlabel('Input voltage (V)')
        ax.set_ylabel(f'{tau_type.capitalize()} Tau (s)')
        ax.set_title(f'{cat}_{tau_type}')
        cat_v = f'charge_{cat[-1]}'
        for i in range(self.trials.shape[0]):
            for j in range(self.trial_data.shape[1]):
                this_trial = self.trials[i,j] #type: TrialPr
                vs = [meas[0] for meas in this_trial.V_in_meas[cat_v]]
                if tau_type == 'current':
                    taus = [tau[0] for tau in this_trial.current_taus[cat]]
                elif tau_type =='voltage':
                    taus = [tau[0] for  tau in this_trial.voltage_taus[cat]]

                ax.scatter(vs, -np.array(taus), color = colors[j], marker = markers[i%5])
                if plot_det:
                    this_trial.current_time_constant_est(cat, True)
        return fig, ax
    def video_file_paths(self):
        """returns video file paths for matlab"""
        return self.metadata['video_fname']
    








 
        



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
    def analyze(self):
        #does a complete analysis of provided trial info
        self.separate_data()
        available_cats = list(self.hr_data.keys())
        self.align_all_to_trigger()
        self.detect_peak_current(available_cats, frac = 0.1, filter = self.well_triggered)
        self.current_time_constant_est(available_cats, False, filter = self.well_triggered)
        self.get_input_voltage(available_cats, [3e-5, self.delay/1000*0.75])
        self.voltage_time_constant_est(['charge_A', 'discharge_A'], filter = self.well_triggered)

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
                trigger_pts = high_indxs[:-1] & low_indxs[1:]
            trigger_pts = np.flatnonzero(trigger_pts)
            zero_time_indx = np.abs(this_trial['time']).argmin()
            if len(trigger_pts)>=1:
                chosen_triggerpt = trigger_pts[np.abs(trigger_pts-zero_time_indx).argmin()]
                new_time = this_trial['time'] - this_trial['time'][chosen_triggerpt]
                this_trial['time_og'] = this_trial['time']
                shifts[ind] = this_trial['time'][chosen_triggerpt]
                this_trial['time'] = new_time
        return data, shifts
    def plot_time(self, data:'np.ndarray[pd.DataFrame]', time_window = None, trigger = 'trigger', filter_trigger = False):
        num_of_trials = len(data)
        fig, ax = plt.subplots(3,1, sharex=True)
        cmap = mpl.cm.get_cmap('viridis')
        colors = cmap(np.linspace(0,1,num_of_trials))
        for ind, trial_data in enumerate(data):
            if time_window is None:
                time_window = np.array([ trial_data['time'][0], trial_data['time'].iloc[-1]])
            if filter_trigger:
                try:
                    trial_data['time_og']
                except:
                    continue
            trial_data = trial_data[ (trial_data['time']>=time_window[0])& (trial_data['time']<=time_window[1])]
            ax[0].plot(trial_data['time']*1000, trial_data['V_dev'], color = colors[ind])
            ax[1].plot(trial_data['time']*1000, trial_data[trigger], color = colors[ind])
            ax[2].plot(trial_data['time']*1000, trial_data['V_rout'], color = colors[ind])
        ax[2].set_xlabel('Time (ms)')
        ax[0].set_ylabel('Device voltage(V)')
        ax[1].set_ylabel('Trigger signal (V)')
        ax[2].set_ylabel('V_rout (V)')
    def align_all_to_trigger(self):
        channel = ['trigger','trigger','trigger_B','trigger_B']
        rising_edge = [False, True, False, True]
        classification = ['charge_A', 'discharge_A', 'charge_B', 'discharge_B']
        self.well_triggered = {}
        for i, tag in enumerate(classification):
            this_set = self.hr_data[tag]
            success = [False] *len(this_set)
            if len(this_set)>0:
                data, _ = self.align_to_trigger(this_set, channel[i], rising_edge[i])
                for i,this_data in enumerate(data):
                    try:
                        this_data['time_og']
                        success[i] = True
                    except:
                        success[i] = False
            self.well_triggered[tag] = success


    # def find_max(self, window = [-1e-5,            
    def detect_peak_current(self,data_category = 'charge_A', time_window = (-1e6,70e-6), frac = 0.1, filter = True):
        """ 
        finds time and value for peak V_rout for a desired block
        output is a list of tuples (argval, time, peak, fracpoint arg, time, value)
        """
        if isinstance(data_category,list):
            results = {}

            for cat in data_category:
                if filter is True:
                    this_filt = True
                else:
                    this_filt = filter[cat]
                results[cat] = self.detect_peak_current(cat, time_window = time_window, frac = frac, filter = this_filt)
            self.current_peaks = results
            return results
        
        else:
            data = self.hr_data[data_category]
            output = []
            if filter == True:
                filter = [True for _ in data]
            for ind, trial_data in enumerate(data):
                if (filter is not None) and (filter[ind]):
                    trial_data = trial_data[(trial_data['time']>=time_window[0])& (trial_data['time']<=time_window[1])]
                    abs_I = np.abs(trial_data['V_rout'])
                    argmax = abs_I.idxmax()
                    max_I = trial_data['V_rout'].loc[argmax]
                    time_point = trial_data['time'].loc[argmax]
                    second_point_val = np.abs(max_I)*frac

                    arg_sp = abs_I.index[abs_I<= second_point_val].to_numpy()
                    after_peak = arg_sp[arg_sp> argmax]
                    if len(after_peak)>0:
                        arg_sp = int(np.min(arg_sp[arg_sp> argmax]))-1
                        time_sp, vrout_sp =  trial_data['time'].loc[arg_sp], trial_data['V_rout'].loc[arg_sp]
                    else:
                        arg_sp, time_sp, vrout_sp = None, None, None
                    output.append((argmax, time_point, max_I, arg_sp, time_sp, vrout_sp))
                else:
                    output.append(tuple([None]*6))
            return output
    def current_time_constant_est(self, data_category = 'charge_A', plot = False,  filter = True):
        """ returns tau, y_intc and res1 for each kind"""
        if isinstance(data_category,list):
            results = {}
            for cat in data_category:
                if filter ==True:
                    this_filt = True
                else:
                    this_filt = filter[cat]
                results[cat] = self.current_time_constant_est(cat, plot = plot, filter = this_filt)
            self.current_taus = results
            return results
        
        else:
            data = self.hr_data[data_category]
            output = []
            if filter == True:
                filter = [True for _ in data]
            if plot:
                cmap = mpl.cm.get_cmap('viridis')
                colors = cmap(np.linspace(0,1,len(data)))
                fig, ax = plt.subplots(2,1, sharex=True)   
                ax[1].set_ylabel('Log V_rout')
                ax[0].set_ylabel('V_rout (V)')
                ax[1].set_xlabel('Time (s)')
            for ind, trial_data in enumerate(data):
                if filter[ind]:
                    range = self.current_peaks[data_category][ind]
                    if range[1] is None or range[4] is None:
                        output.append((None, None, None))
                        continue
                    t_min = range[1] + 0.25*(range[4]-range[1])
                    time_window = (t_min, range[4]) # gets time window values for data
                    trial_data = trial_data[(trial_data['time']>=time_window[0])& (trial_data['time']<=time_window[1])]
                    log_current = np.log(np.abs(trial_data['V_rout']))
                    t = trial_data['time'].values.reshape(-1,1)
                    Amatrix = np.hstack((t, np.ones(t.shape)))
                    res = np.linalg.lstsq(Amatrix, log_current, rcond=None)
                    tau, y_intcp = res[0]
                    if plot:
                        ax[0].plot(trial_data['time'], trial_data['V_rout'], color = colors[ind])
                        ax[1].plot(trial_data['time'], log_current, color = colors[ind])
                        ax[1].plot(t, Amatrix@res[0],'--', color = colors[ind])
                    output.append((1/tau, y_intcp, res[1]))
                else:
                    output.append((None, None, None))
            return output
    def voltage_time_constant_est(self, data_category = 'charge_A', plot = False,  filter = True):
        """ returns tau, y_intc and res1 for each kind"""
        if isinstance(data_category,list):
            results = {}
            for cat in data_category:
                if filter ==True:
                    this_filt = None
                else:
                    this_filt = filter[cat]
                results[cat] = self.voltage_time_constant_est(cat, plot = plot, filter = this_filt)
            self.voltage_taus = results
            return results
        
        else:
            data = self.hr_data[data_category]
            V_in_meas = self.V_in_meas[data_category]
            output = []
            if filter == True:
                filter = [True for _ in data]
            if plot:
                cmap = mpl.cm.get_cmap('viridis')
                colors = cmap(np.linspace(0,1,len(data)))
                fig, ax = plt.subplots(2,1, sharex=True)   
                ax[1].set_ylabel('Log V_dev')
                ax[0].set_ylabel('V_dev (V)')
                ax[1].set_xlabel('Time (s)')
            for ind, trial_data in enumerate(data):
                if filter[ind]:
                    range = self.current_peaks[data_category][ind]
                    if range[1] is None or range[4] is None:
                        output.append((None, None, None))
                        continue
                    t_min = range[1] + 0.25*(range[4]-range[1])
                    t_max = range[4] - 0.25*(range[4]-range[1])
                    time_windows = [(0, t_max)] # gets time window values for data
                    taus = []
                    y_intcps =[]
                    r2s = []
                    for time_window in time_windows:
                        trial_data2 = trial_data[(trial_data['time']>=time_window[0])& (trial_data['time']<=time_window[1])]
                        neg_voltage = V_in_meas[ind][0] - trial_data2['V_dev']
                        log_voltage = np.log(np.abs(neg_voltage))
                        t = trial_data2['time'].values.reshape(-1,1)
                        Amatrix = np.hstack((t, np.ones(t.shape)))
                        res = np.linalg.lstsq(Amatrix, log_voltage, rcond=None)
                        tau, y_intcp = res[0]
                        if plot:
                            ax[0].plot(trial_data2['time'], trial_data2['V_dev'], color = colors[ind])
                            ax[1].plot(trial_data2['time'], log_voltage, color = colors[ind])
                            ax[1].plot(t, Amatrix@res[0],'--', color = colors[ind])
                        taus.append(tau)
                        y_intcps.append(y_intcp)
                        r2s.append(res[0])
                    output.append((1/taus[0], y_intcps[0], r2s[0]))#,1/taus[1], y_intcps[1], r2s[1]))
                else:
                    output.append((None, None, None))
            return output
    def get_input_voltage(self, data_category = 'charge_A', time_window = [3e-5, 1.3e-4]):
        """ finds the input voltage for devices"""
        voltage_channel = 'V_dev'
        if isinstance(data_category,list):
            results = {}
            for cat in data_category:
                # if filter ==True:
                #     this_filt = None
                # else:
                #     this_filt = filter[cat]
                results[cat] = self.get_input_voltage(cat, time_window = time_window)
            self.V_in_meas = results
            return results
        else:
            data = self.hr_data[data_category]
            output = []
            for ind, trial_data in enumerate(data):
                trial_data = trial_data[(trial_data['time']>=time_window[0])& (trial_data['time']<=time_window[1])]
                mean_voltage = np.mean(trial_data[voltage_channel])
                std_voltage = np.std(trial_data[voltage_channel])
                output.append((mean_voltage, std_voltage))
            return output
    def plot_trial_summary(self):
        """Plots all summary FOMS for trial"""
        fig,ax = plt.subplots(2,2)
        fig.suptitle(f'Nom {self.nom_voltage}V, delay = {self.delay} ms')
        x_values = np.arange(0, self.set_size*len(self.hr_data))
        current_taus = [[item[0] for item in val] for _,val in self.current_taus.items()]
        voltage_taus = [[item[0] for item in val] for _,val in self.voltage_taus.items()]
        max_current = [[item[2] for item in val] for _,val in self.current_peaks.items()]
        v_in_mes = [[item[0] for item in val] for _,val in self.V_in_meas.items()]
        v_in_mes_std = [[item[1] for item in val] for _,val in self.V_in_meas.items()]
        current_taus = np.array(current_taus).flatten()
        v_in_mes = np.array(v_in_mes).flatten()
        v_in_mes_std = np.array(v_in_mes_std).flatten()
        voltage_taus = np.array(voltage_taus).flatten()
        max_current = np.array(max_current).flatten()
        
        ax = ax.flatten()
        ax[0].scatter(x_values, current_taus)
        ax[1].scatter(x_values[0:5], voltage_taus)
        ax[2].scatter(x_values, max_current)
        ax[3].errorbar(x_values, v_in_mes, v_in_mes_std, fmt = "o" )







# # 
# # root = Tk()
# # root.withdraw()

# # pickle_file = filedialog.askopenfilenames(filetypes=[("Pickle py files", "*.py")])[0]
# pickle_file = 'C:/Users/mbustamante/Box Sync/Research/Flagellar Motor/Probe tests/20231028/F8F15_B18_E_P1P2P3_220313_data.py'

# # %%
# e_pr = ElectricalDataPr(pickle_file)
# # %%
# e_pr.analyze_all_trials()

# # %%
# e_pr.plot_vin_vs_tau(cat= 'charge_A', tau_type= 'current')

# e_pr.plot_vin_vs_tau(cat= 'discharge_A', tau_type= 'current')

# e_pr.plot_vin_vs_tau(cat='charge_A', tau_type = 'voltage', plot_det = True)
# # e_pr.plot_vin_vs_tau(cat = 'discharge_A', tau_type='voltage')
# # %%
# # e_pr.trials[0,0].V_in_meas
# # e_pr.trial_data.shape

# # # %%
# # this_trial = e_pr.this_trial(3,0)
# # # %%
# # this_trial.separate_data()
# # # %%
# # this_trial.plot_time(this_trial.hr_data['charge_B'])
# # # %%
# # this_trial.plot_time(this_trial.hr_data['charge_B'], [-0.00001, 0.00005])
# # # %%
# # # this_trial.align_to_trigger(this_trial.hr_data['charge_A'], rising_edge=True)
# # # 
# # this_trial.align_all_to_trigger()

# # # %%
# # this_trial.plot_time(this_trial.hr_data['charge_B'])


# #  # %%0
# # e_pr.plot_voltage_nom(errors= False)
# # # %%
# # i=0
# # j=2
# # this_trial = e_pr.this_trial(i,j)
# # # %%
# # this_trial.separate_data()
# # this_trial.align_all_to_trigger()
# # peaks = this_trial.detect_peak_current(['charge_A','discharge_A', 'charge_B'], time_window = [-1e-5,7e-5], frac = 0.1)
# # # %%
# # this_trial.analyze()



# # # %%
# # this_trial.plot_trial_summary()
# # 9# %%

# # %%
