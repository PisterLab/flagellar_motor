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
flow_file = '../../Probe tests/20230927/F1F7_B9_G/143444/converted_videos/F1f7 B9 G 2.500000E+00Ms 40V.npy'
# flow_file = './test.npy'
with open(flow_file, 'rb') as f:
    vel = np.load(f, allow_pickle = True)
# %%
# gray = dense_flow.vel_to_gray(vel, 100)
vel_img = dense_flow.vel_to_img(vel,100 )
gray = vel_img[...,0]
markers, labels, areas, countour =  dense_flow.segment_watershed(vel_img[...,2], True, img = vel_img)
# # %% 
# %%
def compute_velocity(flow : np.ndarray, cap: cv.VideoCapture, 
                     markers: np.ndarray, this_marker = None, pixel_ar = 1, visualize = False):
    """Returns linear and angular 'velocity' from flow field
    Parameters 
    ----------
    flow: flow field, numpy array with shape (height, width, 2, number_of_frames-1)
    markers: segmented image marker
    this_marker: number to select value, if None it should open image and let you click on an area
    pixel_ar: accounts for 
    """
    if visualize:
        plt.imshow(markers == this_marker)
        plt.title('ROI')
        plt.show()
    par = 0.9
    avg_vel = np.zeros((2,flow.shape[3]))
    mask = markers != this_marker
    mask2 = mask[..., None, None] * np.array([True, True])[None, None, :, None]*np.repeat(True, flow.shape[-1])[None, None, None, :]
    filtered_vel = np.ma.masked_array(flow, mask2, dtype = np.float16) ## dimensions are width, height, (x or y), time
    avg_vel = filtered_vel.mean(axis = (0,1)) #average over time 
    indices = np.argwhere(mask == False)
    idx0 = tuple(indices[len(indices)//3])
    vel0 = filtered_vel[idx0[0], idx0[1], :,:]
    dif_vel = filtered_vel - vel0[None, None, ...]
    X,Y = np.meshgrid(range(flow.shape[1]), range(flow.shape[0]))
    difX = X - idx0[1]
    difY = Y - idx0[0]
    p0 = np.moveaxis(np.array((X,Y)), [0,1,2], [2,0,1])
    p1 = p0[..., np.newaxis] + filtered_vel
    frame_count = p1.shape[-1]
    thetas = np.zeros((frame_count,))
    s = np.zeros((frame_count,))
    translation = np.zeros((frame_count,2))
    for index in range(frame_count):
        this_p0 = np.ma.masked_array(p0.reshape((-1,2)))
        this_p1 = p1[...,index]
        this_p1 = this_p1.reshape((-1,2))
        this_p0 = np.ma.masked_array(p0.reshape((-1,2)), mask = this_p1.mask)
        matrix, inliers = cv.estimateAffinePartial2D(
            np.ma.getdata(this_p0)*np.array((par, 1)), np.ma.getdata(this_p1)*np.array((par, 1)),
            None,method = cv.RANSAC, ransacReprojThreshold= 0.25, 
            maxIters=100, confidence=0.95 )
        theta = np.arctan(matrix[1,0]/matrix[1,1])
        s[index] = matrix[0,0]/np.cos(theta)
        translation[index, :] = matrix[...,2]
        thetas[index] = theta

    # print(dif_vel.shape)
    for frame_num in range(flow.shape[3]):
            
    # #     vel = flow[..., frame_num]
    # #     filtered_vel = np.ma.masked_array(vel, mask )
    # #     avg_vel[:, frame_num] = filtered_vel.mean(axis =(0,1))
    # #     vel0 = filtered_vel[(*idx0,[0,1])] #velocity of an arbitrary point with indices idx0
    # #     rel_vel = filtered_vel - vel0
        if visualize and np.abs(vel0[0,frame_num]) > 1:
            index = frame_num
            # this_p0 = np.ma.masked_array()
            this_p1 = p1[...,index]
            # this_p1 = np.swapaxes(this_p1, [0,1,2],[2,0,1]).reshape((2,-1))
            this_p0 = np.ma.masked_array(p0, mask = this_p1.mask)
            this_p1 = np.moveaxis(this_p1, -1, 0).reshape(2,-1)
            this_p0 = np.moveaxis(this_p0, -1, 0).reshape(2,-1)
            matrix, inliers = cv.estimateAffinePartial2D(
                            np.ma.getdata(this_p0).transpose()*np.array((par, 1)), np.ma.getdata(this_p1).transpose()*np.array((par, 1)),
                            None,method = cv.RANSAC, ransacReprojThreshold= 0.25, 
                            maxIters=100, confidence=0.95 )
            theta = np.arctan(matrix[1,0]/matrix[1,1])
            s[index] = matrix[0,0]/np.cos(theta)
            translation[index, :] = matrix[...,2]
            thetas[index] = theta
            print(matrix, theta, np.sum(inliers))
            this_vel =filtered_vel[..., frame_num]
            rel_vel =dif_vel[..., frame_num]
            sparse = 50
            fig, ax = plt.subplots(2,2)
            fig.set_figheight(10)
            fig.set_figwidth(10)
            fig.suptitle(f'frame{frame_num}')
            ax[0,0].quiver(difX[::sparse, ::sparse]*par,difY[::sparse, ::sparse], this_vel[::sparse,::sparse,0]*par, this_vel[::sparse,::sparse,1], scale =100)
            ax[1,0].quiver(rel_vel[::sparse,::sparse,0], rel_vel[::sparse,::sparse,1],scale =100)
            # ax[0,1].scatter(difY[::sparse, ::sparse], -rel_vel[::sparse,::sparse,0]*0.9)
            # ax[0,1].scatter(difX[::sparse, ::sparse]*0.9, rel_vel[::sparse,::sparse,1])
            ax[0,1].scatter(difX*par, rel_vel[...,1], s=1)
            ax[0,1].scatter(difY, -rel_vel[...,0]*par, s=1)
            ax[0,1].plot(difX*par, -difX*par*theta)
            ax[0,1].plot(difY, -difY*theta)
            ax[0,1].scatter([0,0], avg_vel[:, frame_num])
            ax[1,1].scatter(-difY[::sparse, ::sparse], this_vel[::sparse,::sparse,0]*par)
            ax[1,1].scatter(difX[::sparse, ::sparse]*par, this_vel[::sparse,::sparse,1])
            ax[1,1].scatter([0,0], avg_vel[:, frame_num])
            plt.show()

    return avg_vel, dif_vel, difX, difY, thetas

# %%

avg_vel = compute_velocity(vel, None, markers, labels[0], visualize = False)
# %%

