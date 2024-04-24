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
import pyperclip
import re
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
def process_matfile(out, file, eng = None):
    num_delays, num_voltages, num_meas = out['output'].shape

    data = out['output']
    if eng is None:
        eng = matlab.engine.start_matlab()
    # eng = oct2py.Oct2Py()
    eng.eval(f's = load("{file}");', nargout = 0)
    print(file) 
    note = eng.eval('s.note;')
    date = re.findall(r's/(2023\d{4})/', file)[0]
    dev, time = re.findall(r'/([A-Z]\w+?)_(\d{6})',file)[0]
    pyperclip.copy(f'{date}\t{dev}\t{time}\t\t{note}')
    print(date, time, dev, note)
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
                if data[i,j,k].shape == (1,0) or 't' not in data[i,j,k].dtype.names:
                    continue
                time = data[i,j,k]['t'][0][0]
                V_dev = data[i,j,k]['V_dev'][0][0]
                V_rout = data[i,j,k]['V_rout'][0][0]
                trigger = data[i,j,k]['trigger_signal'][0][0]

                o_data[i,j,k] = pd.DataFrame(data = {
                    'time': time[0,:], 'V_dev': V_dev[0,:], 'V_rout': V_rout[0,:], 'trigger': trigger[0,:]
                    })
                try:
                    trigger2 = data[i,j,k]['trigger_signal2'][0][0]
                    o_data[i,j,k]['trigger_B'] = trigger2[0,:]
                except Exception as error:
                    print(error)
                    pass 
                metadata['R_out'][i,j,k] = data[i,j,k]['R_out'][0][0][0,0]
                metadata['V_in'][i,j,k] = data[i,j,k]['V_in'][0][0][0,0]
                if 'delay' in data[i,j,k].dtype.names:
                    metadata['delay'][i,j,k] = data[i,j,k]['delay'][0][0][0,0]
                else:
                    metadata['delay'][i,j,k] = out['delay'][0][i]
                metadata['light_level'][i,j,k] = data[i,j,k]['ligth_level'][0][0][0,0]
                if k ==0:
                    vfile = eng.eval("s.output{{{0},{1},{2}}}.videoLogfile".format(i+1,j+1,k+1))
                    try:
                        print()
                        metadata['video_fname'][i,j] = vfile.split('"')[1] 
                    except:
                        metadata['video_fname'][i,j] = vfile
    return metadata, o_data
# %%
eng = matlab.engine.start_matlab()
# %%
out, files = load_mat()
# %%
# eng = matlab.engine.start_matlab()
for i, this_out in enumerate(out, start =0):
    processed = process_matfile(this_out, files[i], eng)
    path,fname = os.path.split(files[i])
    name = fname.split('.mat')[0]
    output_fname = os.path.join(path, name+'.py')
    with open(output_fname, 'wb') as output_file:
        pickle.dump(processed,output_file)
        output_file.close()
    eng.eval('clear s;', nargout =0)

# %%