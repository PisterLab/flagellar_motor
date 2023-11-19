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

# %% Import device characteristics


# %%

def load_mat():
    root = Tk()
    root.withdraw()
    files = filedialog.askopenfilenames(filetypes=[("Mat files", "*.mat")])
    if not files:
        print("No files selected")
    output = []
    for file in files:
        mat_contents = sio.loadmat(file)
        output.append(mat_contents)
    return output, files


# %% 
def process_matfile(out, file):
    num_delays, num_voltages, num_meas = out['output'].shape

    data = out['output']
    eng = matlab.engine.start_matlab()
    eng.evalc(f's = load("{file}");')
    metadata = {
        'R_out': np.zeros(data.shape),
        'V_in': np.zeros(data.shape),
        'delay': np.zeros(data.shape),
        'light_level': np.zeros(data.shape),
        'video_fname': np.empty(data.shape[0:2], dtype = object),
        'frame_rate': 0
    }
    o_data = np.empty(data.shape, dtype = pd.DataFrame)
    for i in range(num_delays):
        for j in range(num_voltages):
            for k in range(num_meas):
                time = data[i,j,k]['t'][0][0]
                V_dev = data[i,j,k]['V_dev'][0][0]
                V_rout = data[i,j,k]['V_rout'][0][0]
                trigger = data[i,j,k]['trigger_signal'][0][0]
                o_data[i,j,k] = pd.DataFrame(data = {'time': time[0,:], 'V_dev': V_dev[0,:], 'V_rout': V_rout[0,:], 'trigger': trigger[0,:]})
                metadata['R_out'][i,j,k] = data[i,j,k]['R_out'][0][0][0,0]
                metadata['V_in'][i,j,k] = data[i,j,k]['V_in'][0][0][0,0]
                metadata['delay'][i,j,k] = data[i,j,k]['delay'][0][0][0,0]
                metadata['light_level'][i,j,k] = data[i,j,k]['ligth_level'][0][0][0,0]
                if k ==0:
                    vfile = eng.evalc("s.output{{{0},{1},{2}}}.videoLogfile".format(i+1,j+1,k+1))
                    metadata['video_fname'][i,j] = vfile.split('"')[1] 
    return metadata, o_data
# %%
# %%
out, files = load_mat()
# %%
for i, this_out in enumerate(out):
    processed = process_matfile(this_out, files[i])
    path,fname = os.path.split(files[i])
    name = fname.split('.mat')[0]
    output_fname = os.path.join(path, name+'.py')
    with open(output_fname, 'wb') as output_file:
        pickle.dump(processed,output_file)
        output_file.close()


# %%