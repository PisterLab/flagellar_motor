
import cv2 as cv
import numpy as np
import tkinter as tk
from tkinter import filedialog
from math import sqrt, atan2, degrees
import csv
import matplotlib.pyplot as plt


# %%




def main():

    root = tk.Tk()
    root.withdraw()

    video_file = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
    if not video_file:
        print("No video file selected.")
        return

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
    prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
    mask = np.zeros_like(first_frame)
    mask[..., 1] = 255
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv.imshow("input", frame)

    
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prev_gray, gray, 
                                       None,
                                       0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
        # Sets image hue according to the optical flow 
        # direction
        mask[..., 0] = angle * 180 / np.pi / 2
        
        # Sets image value according to the optical flow
        # magnitude (normalized)
        mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
        
        # Converts HSV to RGB (BGR) color representation
        rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
        
        # Opens a new window and displays the output frame
        cv.imshow("dense optical flow", rgb)
        
        # Updates previous frame
        prev_gray = gray
        if cv.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
