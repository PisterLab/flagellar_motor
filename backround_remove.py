# %%


import cv2 as cv
import numpy as np
import tkinter as tk
from tkinter import filedialog
from math import sqrt, atan2, degrees
import csv
import matplotlib.pyplot as plt


# %% 
def colorConvert(image):
  return(cv.cvtColor(image, cv.COLOR_BGR2RGB))
# %% 
root = tk.Tk()
root.withdraw()

video_file = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
if not video_file:
    print("No video file selected.")
    

cap = cv.VideoCapture(video_file)
# %%
frame_get = cap.get(cv.CAP_PROP_FRAME_COUNT) * np.random.uniform(size = 30)
# %%

#Storing captured frames in an array
frames = []
for i in frame_get:
  cap.set(cv.CAP_PROP_POS_FRAMES, i)
  ret, frame = cap.read()
  frames.append(frame)

cap.release()

# %%

frame_median = np.median(frames, axis = 0).astype(dtype = np.uint8)
plt.imshow(colorConvert(frame_median))
# %%

frame_avg = np.average(frames, axis = 0).astype(dtype = np.uint8)
plt.imshow(colorConvert(frame_avg))
# %%
gray_frame_median = cv.cvtColor(frame_median, cv.COLOR_BGR2GRAY)
plt.imshow(colorConvert(gray_frame_median))
     
# %%


gray_frame_sample = cv.cvtColor(frame_sample, cv2.COLOR_BGR2GRAY)
plt.imshow(colorConvert(gray_frame_sample))
# %%
bg_removed_frame = cv.absdiff(gray_frame_sample, gray_frame_median)
plt.imshow(colorConvert(bg_removed_frame))
     

# %%
