# %%
import cv2 as cv
import numpy as np
import tkinter as tk
from tkinter import filedialog
from math import sqrt, atan2, degrees
import csv
import dense_flow
import os 
import matplotlib.pyplot as plt
# %% 
# video_file = os.path.abspath('./test.mp4')
# video_file = './test.mp4'


# %%
# vel, pixel_ar = dense_flow.dense_flow_on_video(video_file)

# %%
# flow_file = '../../Probe tests/20220222/E9V5.1h/1 50V 250Ms 2023-02-22-213053-0004.npy'
flow_file = './test.npy'
with open(flow_file, 'rb') as f:
    vel = np.load(f, allow_pickle = True)
# %%
# gray = dense_flow.vel_to_gray(vel, 100)
vel_img = dense_flow.vel_to_img(vel,100 )
gray = vel_img[...,0]
markers, labels, areas, countour =  dense_flow.segment_watershed(vel_img[...,2], True, img = vel_img)
# # %% 