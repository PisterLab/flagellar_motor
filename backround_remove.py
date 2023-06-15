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
# %%
def get_motion_ROI(cap: cv.VideoCapture, visualize = False):
  """Returns an region of interest ROI to look for good points to track by removing video average """
  frame_get = cap.get(cv.CAP_PROP_FRAME_COUNT) * np.random.uniform(size = 30)
  frames = []
  for i in frame_get:
    cap.set(cv.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    frames.append(frame)
  cap.release()
  frame_avg = np.average(frames, axis = 0).astype(dtype = np.uint8)
  gray_frame_avg = cv.cvtColor(frame_avg, cv.COLOR_BGR2GRAY)
  frame_sample = frames[0]
  gray_frame_sample = cv.cvtColor(frame_sample, cv.COLOR_BGR2GRAY)
  bg_removed_frame = cv.absdiff(gray_frame_sample, gray_frame_avg)

  ret,th = cv.threshold(bg_removed_frame,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
  ret2,th2 = cv.threshold(bg_removed_frame,ret*0.9,255, cv.THRESH_BINARY)
  kernel = np.ones((5,5),np.uint8)
  dilated = cv.dilate(th2, kernel, iterations=5)
  dilated = cv.morphologyEx(dilated, cv.MORPH_OPEN, kernel)
  if visualize:
    plt.imshow(colorConvert(bg_removed_frame))
    plt.imshow(colorConvert(dilated))
  return dilated



# %%
def main():
  root = tk.Tk()
  root.withdraw()
  video_file = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
  if not video_file:
      print("No video file selected.")
  cap = cv.VideoCapture(video_file)
  roi = get_motion_ROI(cap, True)


# %%
if __name__ == '__main__':
  main()