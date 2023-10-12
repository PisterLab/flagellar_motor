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

def dense_flow_on_video(video_file):
    cap = cv.VideoCapture(video_file)
    numerator = cap.get(cv.CAP_PROP_SAR_NUM) or 1.0
    denominator = cap.get(cv.CAP_PROP_SAR_DEN) or 1.0
    pixel_ar = numerator / denominator
    _, first_frame = cap.read()
    prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    vel = np.zeros((*prev_gray.shape,2,frame_count), dtype = np.float16)
    flow = np.zeros((*prev_gray.shape,2), dtype = np.float32)
    for frame_ind in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            print('reach_end')
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        cv.calcOpticalFlowFarneback(prev_gray, gray, 
                                       flow,
                                       0.5, 4, 30, 4, 5, 1.1, 0) 
        vel[...,frame_ind] = flow  
        prev_gray = gray 
    cap.release()
    return vel, pixel_ar
def imshow(img, ax=None):
    if ax is None:
        ret, encoded = cv.imencode(".jpg", img)
        display(Image(encoded))
    else:
        ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        ax.axis('off')

def vel_to_gray(vel:np.ndarray, percentile=99):
    if percentile == 100:
        vel_per = np.max(vel, axis =3)
    else:
        vel_per = np.percentile(vel, percentile, axis =3)
    magnitude, angle = cv.cartToPolar(vel_per[..., 0].astype(np.float32), vel_per[...,1].astype(np.float32))
    # sum_vel = np.sum(vel.astype(np.float32), axis =3)
    # magnitude, angle = cv.cartToPolar(sum_vel[..., 0], sum_vel[...,1])
    gray = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
    gray = gray.astype(np.uint8)
    return gray
def vel_to_img(vel:np.ndarray, percentile=99):
    mask = np.zeros((*vel.shape[0:2],3), dtype = np.uint8)
    if percentile == 100:
        vel_per = np.max(vel, axis =3)
    else:
        vel_per = np.percentile(vel, percentile, axis =3)
    magnitude, angle = cv.cartToPolar(vel_per[..., 0].astype(np.float32), vel_per[...,1].astype(np.float32))
    # sum_vel = np.sum(vel.astype(np.float32), axis =3)
    # magnitude, angle = cv.cartToPolar(sum_vel[..., 0], sum_vel[...,1])
    mask[...,2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
    mask[..., 0] = angle * 180 / np.pi / 2
    mask[...,1] = 255
    # gray = gray.astype(np.uint8)
    return mask

def segment_watershed(gray, display=False, img = None):
    """Takes a gray scale image and runs watershed. If display is true shows progress in graphs
    Parameters
    ----------
    gray: grayscale image
    display: boolean, display progress"""

    blur = cv.GaussianBlur(gray,(5,5),0)
    ret, bin_img = cv.threshold(gray,
                            0, 255, 
                             cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    bin_img = 255-bin_img #invert image
    #remove noise
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    bin_img = cv.morphologyEx(bin_img, 
                           cv.MORPH_OPEN,
                           kernel,
                           iterations=3)
    
    #sure
    sure_bg = cv.dilate(bin_img, kernel, iterations=3)
    # 
    # Distance transform
    dist = cv.distanceTransform(bin_img, cv.DIST_L2, 5)
    ret, sure_fg = cv.threshold(dist, 0.5*np.std(dist), 255, cv.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)      
    unknown = cv.subtract(sure_bg, sure_fg)
    ret, markers = cv.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 255] = 0
    if display:
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
        imshow(sure_bg, axes[0,0])
        axes[0, 0].set_title('Sure Background') 
        imshow(dist, axes[0,1])
        axes[0, 1].set_title('Distance Transform')
        imshow(sure_fg, axes[1,0])
        axes[1, 0].set_title('Sure Foreground')
        imshow(unknown, axes[1,1])
        axes[1, 1].set_title('Unknown')
        plt.show()

    # watershed 
    if img is None:
        img = cv.cvtColor(255-gray, cv.COLOR_GRAY2BGR)
    else:
        img = cv.cvtColor(img, cv.COLOR_HSV2BGR)
    markers = cv.watershed(img, markers)
    labels = np.unique(markers)
    coins = []
    areas = []
    for label in labels[2:]:  

        # Create a binary image in which only the area of the label is in the foreground 
        #and the rest of the image is in the background   
        target = np.where(markers == label, 255, 0).astype(np.uint8)
        area = np.sum(target)/255
                # Perform contour extraction on the created binary image
        contours, hierarchy = cv.findContours(
            target, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
            )
        coins.append(contours[0])
        areas.append(area)
    args = np.flip(np.argsort(areas))
    out_labels = labels[2:][args]
    out_areas = np.array(areas)[args.astype(int)]

    # Draw the outline
    imgc = cv.drawContours(img, coins, -1, color=(0, 23, 223), thickness=2)
    if display: 
        imshow(imgc)
    return markers, out_labels, out_areas, imgc


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
                                       0.5, 3, 15, 3, 5, 1.1, 0)
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

# %%
