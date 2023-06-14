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

# # Plot the image
# from IPython.display import Image, display
# def imshow(img, ax=None):
#     if ax is None:
#         ret, encoded = cv.imencode(".jpg", img)
#         display(Image(encoded))
#     else:
#         ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
#         ax.axis('off')


# # %%

# #Threshold Processing
# ret, bin_img = cv.threshold(gray,
#                             0, 255, 
#                              cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
# bin_img = 255-bin_img
# imshow(bin_img)


# # %% 

# # noise removal
# kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
# bin_img = cv.morphologyEx(bin_img, 
#                            cv.MORPH_OPEN,
#                            kernel,
#                            iterations=3)
# imshow(bin_img)

# # %% 
# # Create subplots with 1 row and 2 columns
# cv2 = cv
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
# # sure background area
# sure_bg = cv2.dilate(bin_img, kernel, iterations=3)
# imshow(sure_bg, axes[0,0])
# axes[0, 0].set_title('Sure Background')
  
# # Distance transform
# dist = cv2.distanceTransform(bin_img, cv2.DIST_L2, 5)
# imshow(dist, axes[0,1])
# axes[0, 1].set_title('Distance Transform')
  
# #foreground area
# ret, sure_fg = cv2.threshold(dist, 0.5*np.std(dist), 255, cv2.THRESH_BINARY)
# sure_fg = sure_fg.astype(np.uint8)  
# imshow(sure_fg, axes[1,0])
# axes[1, 0].set_title('Sure Foreground')
  
# # unknown area
# unknown = cv2.subtract(sure_bg, sure_fg)
# imshow(unknown, axes[1,1])
# axes[1, 1].set_title('Unknown')
  
# plt.show()


# # %%
# # Marker labelling
# # sure foreground 
# ret, markers = cv2.connectedComponents(sure_fg)
  
# # Add one to all labels so that background is not 0, but 1
# markers += 1
# # mark the region of unknown with zero
# markers[unknown == 255] = 0
  
# fig, ax = plt.subplots(figsize=(6, 6))
# ax.imshow(markers, cmap="tab20b")
# ax.axis('off')
# plt.show()

# # %% watershed Algorithm
# img = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
# markers = cv2.watershed(img, markers)
  
# fig, ax = plt.subplots(figsize=(5, 5))
# ax.imshow(markers, cmap="tab20b")
# ax.axis('off')
# plt.show()
  
  
# labels = np.unique(markers)
  
# coins = []
# for label in labels[2:]:  
  
# # Create a binary image in which only the area of the label is in the foreground 
# #and the rest of the image is in the background   
#     target = np.where(markers == label, 255, 0).astype(np.uint8)
#     if np.sum(target)/label> 1:
#         imshow(target)
#         plt.show()
#          # Perform contour extraction on the created binary image
#         contours, hierarchy = cv2.findContours(
#         target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
#         )
#         coins.append(contours[0])
# # Draw the outline
# imgc = cv2.drawContours(img, coins, -1, color=(0, 23, 223), thickness=2)
# imshow(imgc)
# %%
dense_flow.main()
# %%
