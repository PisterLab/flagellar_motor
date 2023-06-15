# %%

import cv2 as cv
import numpy as np
import tkinter as tk
from tkinter import filedialog
from math import sqrt, atan2, degrees
import csv
import matplotlib.pyplot as plt
from IPython.display import Image, display

# %%
    
video_file = './test.mp4'

# %%
def imshow(img, ax=None):
    if ax is None:
        ret, encoded = cv.imencode(".jpg", img)
        display(Image(encoded))
    else:
        ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        ax.axis('off')

# %%
cap = cv.VideoCapture(video_file)
storage_w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
storage_h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)) 
sar = storage_w / storage_h
print(f"Storage aspect ratio: {sar:.2f} (resolution: {storage_w}×{storage_h})")

numerator = cap.get(cv.CAP_PROP_SAR_NUM) or 1.0
denominator = cap.get(cv.CAP_PROP_SAR_DEN) or 1.0
par = numerator / denominator
print(f"Pixel aspect ratio: {par:.2f} (resolution: {numerator}×{denominator})")
_, first_frame = cap.read()
print(first_frame.shape)

# %%
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
# %%
img = first_frame.copy()
corners = cv.goodFeaturesToTrack(prev_gray,100,0.0001,50)
corners = np.int0(corners)
for i in corners:
    x,y = i.ravel()
    cv.circle(img,(x,y),10,255,-1)
imshow(img)
# %%
def main():
    root = tk.Tk()
    root.withdraw()

    video_files = filedialog.askopenfilenames(filetypes=[("MP4 files", "*.mp4")])
    if not video_files:
        print("No video file selected.")
        return
    for video_file in video_files:
        directory = os.path.dirname(video_file)
        sparse_flow_videofile(video)
        # flow, _ = dense_flow.dense_flow_on_video(video_file)
        # basename_without_ext = os.path.splitext(os.path.basename(video_file))[0]
        # write_fname = os.path.join(directory, basename_without_ext+'.npy')
        # with open(write_fname, "wb") as f:
        #     np.save(f, flow.astype(np.float16))
if __name__ == '__main__':
    main()
# %%
