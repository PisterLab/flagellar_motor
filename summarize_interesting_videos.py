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

pickle_file = 'C:/Users/mbustamante/Box Sync/Research/Flagellar Motor/Probe tests/20231028/F8F15_B18_E_P1P2P3_213679_data.py'

pr = JoinProcessor(pickle_file)
pr.analyze_video_data(correction_factor=30/42.15)

# %%
pr.combine_data()

# %%
pr.voltage_vs_avg_speed()


# %% 
pr.plot('meas_voltages', 'speeds', 'times')
pr.plot('times', 'speeds', 'nom_voltages')
pr.plot_with_colorbar('frequency', 'med_step_sizes', 'meas_voltages')
pr.plot_with_colorbar('frequency', 'speeds', 'meas_voltages')
# %%
pr.Elec.plot_voltage_nom('charge_A')

# %% 
pickle_file = 'C:/Users/mbustamante/Box Sync/Research/Flagellar Motor/Probe tests/20231028/F8F15_B18_E_P1P2P3_213679_data.py'
