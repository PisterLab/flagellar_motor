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
gray_frame_avg = cv.cvtColor(frame_avg, cv.COLOR_BGR2GRAY)
plt.imshow(colorConvert(frame_avg))
# %%
gray_frame_median = cv.cvtColor(frame_median, cv.COLOR_BGR2GRAY)
plt.imshow(colorConvert(gray_frame_median))
     
# %%
frame_sample = frames[0]

gray_frame_sample = cv.cvtColor(frame_sample, cv.COLOR_BGR2GRAY)
plt.imshow(colorConvert(gray_frame_sample))
# %%
bg_removed_frame = cv.absdiff(gray_frame_sample, gray_frame_avg)
plt.imshow(colorConvert(bg_removed_frame))
# %%


ret,th = cv.threshold(bg_removed_frame,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
ret2,th2 = cv.threshold(bg_removed_frame,ret*0.9,255, cv.THRESH_BINARY)
kernel = np.ones((5,5),np.uint8)
dilated = cv.dilate(th2, kernel, iterations=5)
dilated = cv.morphologyEx(dilated, cv.MORPH_OPEN, kernel)
plt.imshow(colorConvert(th2))
plt.imshow(colorConvert(dilated))
# %%
bg_rmoved_color = cv.absdiff(frame_sample, frame_avg)
plt.imshow(colorConvert(bg_rmoved_color*10))

# %% threschold 
# %%
# cap = cv.VideoCapture(video_file)
# backSub = cv.createBackgroundSubtractorMOG2()
# while True:
#     ret, frame = cap.read()
#     if frame is None:
#         break
    
#     fgMask = backSub.apply(frame)
    
    
#     cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
#     cv.putText(frame, str(cap.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
#                cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    
    
#     cv.imshow('Frame', frame)
#     cv.imshow('FG Mask', fgMask)
    
#     keyboard = cv.waitKey(30)
#     if keyboard == 'q' or keyboard == 27:
#         break
# %%
