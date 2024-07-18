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
from elec_and_optical_processing import *
# %% 

# pickle_file = 'C:/Users/mbustamante/Box Sync/Research/Flagellar Motor/Probe tests/20231028/F8F15_B18_E_P1P2P3_213679_data.py'
# pickle_file = os.path.join(pickle_file)
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
# pickle_file = r"C:\Users\mbustamante\Box Sync\Research\Flagellar Motor\Probe tests\20230927\F1F7_B9_B__150594_data.py"
# pickle_file = os.path.join(pickle_file)
# # %%
# pr = JoinProcessor(pickle_file)
# pr.analyze_video_data(correction_factor=30/42.15)

# %%
pickle_file = r"C:\Users\mbustamante\Box Sync\Research\Flagellar Motor\Probe tests\20231028\F8F15_B18_C_P1P2 not P3_200780_data.py"
pickle_file = os.path.join(pickle_file)
pr = JoinProcessor(pickle_file)
pr.analyze_video_data(correction_factor=30/42.15)
# %%
pr.combine_data()
# %%
pr.plot_with_colorbar('frequency', 'speeds','nom_voltages')
# %%
pr.voltage_vs_avg_speed()
# %%
pr.speeds[-1] = 5.45*42.15
pr.speeds[-2] = 2.44*42.15
pr.plot_with_colorbar('frequency', 'speeds','nom_voltages')
# %%
pr.plot('frequency', 'speeds','nom_voltages', log = 'loglog')
# %%
pr.Elec.trials[-1][0].plot_time(pr.Elec.trial_data[-1][0], time_window = np.array([-10e-3,10e-3]))
pr.Elec.trials[-2][0].plot_time(pr.Elec.trial_data[-2][0], time_window = np.array([-10e-3,10e-3]), filter_trigger= True)

# %%
pr.Elec.trial_data[-1]
# %%
plt.plot(pr.Elec.trial_data[-1][0][0].time, pr.Elec.trial_data[-1][0][0].trigger)
# %%
plt.plot(pr.Elec.trial_data[-2][0][0].time, pr.Elec.trial_data[-2][0][0].trigger)

# %%
# %%
pickle_file=  r"C:\Users\mbustamante\Box Sync\Research\Flagellar Motor\Probe tests\20231028\F8F15_B18_C_P1P2 not P3_200298_data.py"
pr2 = JoinProcessor(pickle_file)
# %%

pr2.analyze_video_data()
pr2.combine_data()
# %%
pr2.plot('frequency', 'speeds', 'nom_voltages')
# %%
pr2.speeds[-1,0] = 2.9605*42.15
# %%
pr2.plot('frequency', 'speeds', 'nom_voltages')
# %%
pr3 = pr+pr2
# %%
pr3.plot_with_colorbar('frequency', 'speeds', 'meas_voltages', log = 'loglog')
# %%
trial_pr = pr3.Elec.trials[9,0]

# %%
