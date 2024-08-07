""" Join electrical data with optical processing"""
# %%

import numpy as np
from IPython.display import Image, display
import os
import pandas as pd
import matlab.engine
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl

import matlab_data_pr as elec
import process_angular_data as opt
from copy import copy

# %%
class JoinProcessor():
    def __init__(self,  pickled_file_path):
        self.pickled_fpath = pickled_file_path
        self.Elec = elec.ElectricalDataPr(pickled_file_path)
        video_fpath = os.path.split(self.Elec.video_file_paths().flatten()[0])
        temp_path = video_fpath[0]
        tail = video_fpath[1]
        fname_comps = tail.split('_')[0:3]
        csv_tail = '_'.join(fname_comps) + '.csv'
        fpath_comps = temp_path.split(os.sep)[1:]
        elec_fpath = os.path.split(self.pickled_fpath)[0]
        self.csv_fpath = os.path.join(elec_fpath, *fpath_comps, csv_tail)
        self.Opti = opt.AngularDataPr(self.csv_fpath)
        self.Elec.analyze_all_trials()
    def analyze_video_data(self, correction_factor =30/42.15, visualize = True ):

        self.Opti.process_all(time_correction_factor= correction_factor, visualize=visualize)
        self.Opti.save_FOMS()
        # self.motion_csv_fpath = os.path.join(os.path.split(self.pickled_fpath)[0], os.path.split(self.Elec)

    def combine_data(self):
        """ makes a matrix combining all the FOMs for the trials provided"""
        data = self.Opti.summary_data
        data_shape = self.Elec.trial_data.shape[0:2]
        nom_voltages = np.empty(data_shape)
        meas_voltages = np.empty(data_shape)
        times = np.empty(data_shape)
        speeds = np.empty(data_shape)
        avg_step_sizes = np.empty(data_shape)
        med_step_sizes = np.empty(data_shape)
        med_step_period = np.empty(data_shape)
        current_taus = {
            'charge_A': np.empty((*data_shape, 5)),
            'charge_B': np.empty((*data_shape, 5)),
            'discharge_A': np.empty((*data_shape, 5)),
            'discharge_B': np.empty((*data_shape, 5)),
        }
        voltage_taus = {
            'charge_A': np.empty((*data_shape, 5)),
            'discharge_A': np.empty((*data_shape, 5))
        }
        for i in range(self.Elec.trial_data.shape[0]):
            for j in range(self.Elec.trial_data.shape[1]):
                this_trial = self.Elec.trials[i,j] #type: elec.TrialPr
                # this_trial.plot_time(this_trial.hr_data['charge_A'], time_window=[-0.00001, 0.00005])
                nom_voltages[i,j] = this_trial.nom_voltage
                meas_voltage = [meas[0] for meas in this_trial.V_in_meas['charge_A']]
                meas_voltages[i,j] = np.mean(np.array(meas_voltage))
                times[i,j] = this_trial.delay
                filtered_data = data[(data['delay_time'] == this_trial.delay) & (data['nom_voltage'] == this_trial.nom_voltage)]
                if not len(filtered_data) == 1:
                    print('Error finding exact data\n', filtered_data)  
                speeds[i,j] = filtered_data['average_speed']
                avg_step_sizes[i,j] = filtered_data['avg_step_size']
                med_step_sizes[i,j] = filtered_data['median_step_size']
                med_step_period[i,j] = filtered_data['median_step_period']
                for cat in ['charge_A', 'discharge_A']:
                    current_taus[cat][i,j,:] = np.array([tau[0] for tau in this_trial.current_taus[cat]])
                    voltage_taus[cat][i,j, :] = np.array([tau[0] for tau in this_trial.voltage_taus[cat]])
                for cat in ['charge_B', 'discharge_B']:
                    current_taus[cat] = np.array([tau[0] for tau in this_trial.current_taus[cat]])

        self.nom_voltages = nom_voltages
        self.meas_voltages = meas_voltages
        self.times = times
        self.speeds = speeds
        self.avg_step_sizes = avg_step_sizes
        self.med_step_sizes = med_step_sizes
        self.med_step_periods = med_step_period
        self.current_taus = current_taus
        self.voltage_taus = voltage_taus

    def voltage_vs_avg_speed(self):
        data = self.Opti.summary_data
        data_shape = self.Elec.trial_data.shape[0:2]
        nom_voltages = np.empty(data_shape)
        meas_voltages = np.empty(data_shape)
        times = np.empty(data_shape)
        speeds = np.empty(data_shape)
        fig, ax = plt.subplots(1,2)
        cmap = plt.get_cmap('viridis')
        colors = cmap(np.linspace(0,1,self.Elec.trial_data.shape[0]))
        handles = []
        for i in range(self.Elec.trial_data.shape[0]):
            for j in range(self.Elec.trial_data.shape[1]):
                this_trial = self.Elec.trials[i,j] #type: elec.TrialPr
                # this_trial.plot_time(this_trial.hr_data['charge_A'], time_window=[-0.00001, 0.00005])
                nom_voltages[i,j] = this_trial.nom_voltage
                meas_voltage = [meas[0] for meas in this_trial.V_in_meas['charge_A']]
                meas_voltages[i,j] = np.mean(np.array(meas_voltage))
                times[i,j] = this_trial.delay
                filtered_data = data[(data['delay_time'] == this_trial.delay) & (data['nom_voltage'] == this_trial.nom_voltage)]
                if not len(filtered_data) == 1:
                    print('Error finding exact data\n', filtered_data)  
                speeds[i,j] = filtered_data['average_speed']
            handles.append(ax[0].scatter(meas_voltages[i,:], speeds[i,:], color = colors[i]))
            ax[1].scatter(nom_voltages[i,:], speeds[i,:], color = colors[i])
        ax[0].set_xlabel('Measured Input Voltage (V)')
        ax[0].set_ylabel('Speed (deg/s)')
        ax[1].set_xlabel('Nominal Input Voltage (V)')
        ax[1].set_ylabel('Speed (deg/s)')
        legend_ax = ax[1] #type plt.Axes
        legend = sorted(zip(times[:,0], handles))
        legend = list(zip(*legend))
        fig.legend(legend[1], legend[0], title = 'Delay time (ms)')

    def plot(self, xaxis, yaxis, legend, log = False):
        fig, ax = plt.subplots()
        if xaxis == 'times' or xaxis =='frequency':
            x_dim = 0
            if xaxis == 'times':
                x_var = self.times
            if xaxis == 'frequency':
                x_var = 1/(4*self.times*1e3)
            
        elif xaxis == 'meas_voltages' or xaxis == 'nom_voltages':
            x_dim = 1
            x_var = getattr(self, xaxis)

        leg_dim = (x_dim +1)%2
        cmap = plt.get_cmap('viridis')
        colors = cmap(np.linspace(0,1,self.Elec.trial_data.shape[leg_dim]))
        y_var = getattr(self, yaxis)
        leg_var = getattr(self, legend)
        if leg_dim ==1:
            x_var = x_var.swapaxes(0,1)
            y_var = y_var.swapaxes(0,1)
            leg_var = leg_var.swapaxes(0,1)
        handles = []
        index_order = np.argsort(leg_var[:,0])
        for ind in index_order:
            if not log:
                x_plot = x_var[ind,:]
                y_plot = y_var[ind,:]
                append_x = ''
                append_y = ''
            elif log == 'loglog':
                x_plot = np.log10(x_var[ind,:])
                y_plot = np.log10(y_var[ind,:])
                append_x = 'log '
                append_y = 'log '
            elif log == 'semilog':
                x_plot = np.log10(x_var[ind,:])
                y_plot = y_var[ind,:]
                append_x = 'log '
                append_y = ''
            handles.append(ax.scatter(x_plot, y_plot, color =colors[ind]))

        ax.set_xlabel(f'{append_x}{xaxis}')
        ax.set_ylabel(f'{append_y}{yaxis}')
        ax.legend(handles, leg_var[index_order, 0], title = legend)
        return fig, ax 
    def plot_with_colorbar(self, xaxis, yaxis, coloraxis, log = False):
        fig, ax = plt.subplots()
        if xaxis == 'times' or xaxis =='frequency':
            x_dim = 0
            if xaxis == 'times':
                x_var = self.times
            if xaxis == 'frequency':
                x_var = 1/(4*self.times)*1e3 
            
        elif xaxis == 'meas_voltages' or xaxis == 'nom_voltages':
            x_dim = 1
            x_var = getattr(self, xaxis)

        cmap = plt.get_cmap('viridis')
        y_var = getattr(self, yaxis)
        leg_var = getattr(self, coloraxis)
        if not log:
            x_plot = x_var
            y_plot = y_var
            append_x = ''
            append_y = ''
        elif log == 'loglog':
            x_plot = np.log10(x_var)
            y_plot = np.log10(y_var)
            append_x = 'log '
            append_y = 'log '
        elif log == 'semilog':
            x_plot = np.log10(x_var)
            y_plot = y_var
            append_x = 'log '
            append_y = ''
        handles = ax.scatter(x_plot, y_plot, c = leg_var, cmap = cmap, label = coloraxis )
        ax.set_xlabel(append_x + xaxis)
        ax.set_ylabel(append_y + yaxis)
        plt.colorbar(handles)
    def __add__(self, pr2):
            new_pr = copy(self)
            attr_list = ['speeds', 'times', 'meas_voltages', 'nom_voltages', 'med_step_periods', 'med_step_sizes', 
            'avg_step_sizes', ]
            for elemn in attr_list:
                val = np.concatenate([getattr(self, elemn), getattr(pr2, elemn)], axis =0)
                setattr(new_pr, elemn, val)
                # new_pr.speeds = np.concatenate([self.speeds, pr2.speeds], axis =0)
                # new_pr.times = np.concatenate([self.times, pr2.times], axis =0)
            dict_attr_list = ['voltage_taus', 'current_taus']
            for elemn in dict_attr_list:
                attr1 = getattr(self, elemn)#type: dict 
                attr2 = getattr(pr2, elemn) #type: dict
                attr = copy(attr1)
                for key in attr1.keys():
                    attr[key] =  np.concatenate([attr1[key], attr2[key]], axis=0)
                setattr(new_pr, elemn, attr)
                new_pr.Elec = self.Elec+ pr2.Elec
            new_pr.Opti = [self.Opti, pr2.Opti]
        
            new_pr.subprs = [self, pr2]
            return new_pr

# # %% 

# pickle_file = 'C:/Users/mbustamante/Box Sync/Research/Flagellar Motor/Probe tests/20231028/F8F15_B18_E_P1P2P3_213679_data.py'

# pr = JoinProcessor(pickle_file)
# pr.analyze_video_data(correction_factor=30/42.15)

# # %%
# pr.combine_data()

# # %%
# pr.voltage_vs_avg_speed()


# # %% 
# pr.plot('meas_voltages', 'speeds', 'times')
# pr.plot('times', 'speeds', 'nom_voltages')
# pr.plot_with_colorbar('frequency', 'med_step_sizes', 'meas_voltages')
# pr.plot_with_colorbar('frequency', 'speeds', 'meas_voltages')
# # %%
# pr.Elec.plot_voltage_nom('charge_A')
# %%
# def join_two_processors(pr1: JoinProcessor, pr2:JoinProcessor):
#     new_pr = copy(pr1)
#     new_pr.speeds = np.concatenate(pr1.speeds, pr2.speeds)
#     new_pr.times = np.concatenate(pr1.times, pr2.times)
#     for key in new_pr.voltage_taus.keys():
#         new_pr.voltage_taus[key] = np.concatenate(pr1.voltage_taus[key], pr2.voltage_taus[key], axis=0)
#     new_pr.Elec = [pr1.Elec, pr2.Elec]
#     new_pr.Opti = [pr1.Opti, pr2.Opti]
#     return new_pr