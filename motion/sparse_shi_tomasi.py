# %%

import cv2 as cv
import numpy as np
import tkinter as tk
from tkinter import filedialog
from math import sqrt, atan2, degrees
import csv
import matplotlib.pyplot as plt
from IPython.display import Image, display
import os
import pandas as pd

# %%
    
video_file ='../../Probe tests/20230927/F1F7_B9_G/143444/converted_videos/F1f7 B9 G 5Ms 40V.mp4'

# %%
def imshow(img, ax=None):
    if ax is None:
        ret, encoded = cv.imencode(".jpg", img)
        display(Image(encoded))
    else:
        ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        ax.axis('off')

# %%
def process_video_sparse(video_file):
    cap = cv.VideoCapture(video_file)
    storage_w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    storage_h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)) 
    sar = storage_w / storage_h
    frame_rate = cap.get(cv.CAP_PROP_FPS)
    print(f"Storage aspect ratio: {sar:.2f} (resolution: {storage_w}×{storage_h})")

    numerator = cap.get(cv.CAP_PROP_SAR_NUM) or 1.0
    denominator = cap.get(cv.CAP_PROP_SAR_DEN) or 1.0
    par = numerator / denominator
    print(f"Pixel aspect ratio: {par:.2f} (resolution: {numerator}×{denominator})")
    _, first_frame = cap.read()
    print(first_frame.shape)

    prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
    img = first_frame.copy()
    num_features = 1000
    corners = cv.goodFeaturesToTrack(prev_gray,num_features,0.001,10)
    corners = np.int0(corners)
    # for i in corners:
    #     x,y = i.ravel()
    #     cv.circle(img,(x,y),10,255,-1)
    # imshow(img)


    # img = first_frame.copy()
    # # mask = background_remove.get_motion_ROI(cap, True)
    # r = cv.selectROI("select roi", img)
    # mask = np.zeros_like(prev_gray)
    # mask[int(r[1]):int(r[1]+r[3]), 
    #                     int(r[0]):int(r[0]+r[2])] = 1
    # cv.destroyAllWindows()
    # num_features = 1000
    # distance = 0.5* np.sqrt(r[3]**2 + r[2]**2)
    # corners = cv.goodFeaturesToTrack(prev_gray,num_features,0.001,10, mask= mask)
    corners = np.int0(corners)

    # distance = int((r[2]*r[3]/num_features)**0.5)

    # X,Y = np.meshgrid(range(r[0], r[2]+r[0], distance), (range(r[1],r[1]+r[3], distance)))
    # corners = np.array((X.flatten(), Y.flatten())).transpose()
    # corners = corners[:, np.newaxis, :]
    for i in corners:
        x,y = i.ravel()
        cv.circle(img,(x,y),10,255,-1)
    imshow(img)
    cap.release()

    cap = cv.VideoCapture(video_file)

    ret, old_frame = cap.read()
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    p0 = np.float32(corners)
    lk_params = dict( winSize  = (30, 30),
                  maxLevel = 15,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,  50, 0.005))
    # Create some random colors
    color = np.random.randint(0, 255, (num_features, 3))
    color2 = color[:,[2,0,1]]/255

    # Create a mask image for drawing purposes
    lines = np.zeros_like(old_frame)
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    thetas = np.zeros((frame_count,))
    s = np.zeros((frame_count,))
    translation = np.zeros((frame_count,2))
    while(1):
        index = int(cap.get(cv.CAP_PROP_POS_FRAMES))
        ret, frame = cap.read()
        if not ret:
            print('No frames grabbed!')
            break
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]
        else:
            print("None Returned")
        # draw the tracks
        idx0 = p1.shape[0]//2

        vel = p1- p0
        
        idx0 = np.argsort(np.linalg.norm(vel, 2, axis = 2)[:,0])[vel.shape[0]//2]
        vel0 = vel[idx0,...]
        dvel = vel - vel0
        dp0 = p0 - p0[idx0,...]
        dp1 = p1 - p1[idx0,...]
        print(np.linalg.norm(vel0,2))
        matrix, inliers = cv.estimateAffinePartial2D(p0*np.array((par, 1)), p1*np.array((par, 1)), None,method = cv.RANSAC, ransacReprojThreshold= 0.25,  maxIters=100, confidence=0.95 )
        theta = np.arctan(matrix[1,0]/matrix[1,1])
        s[index] = matrix[0,0]/np.cos(theta)
        translation[index, :] = matrix[...,2]
        thetas[index] = theta
        if np.linalg.norm(vel0,2)>0.3:
            print(index)
            dp = p0 - p0[idx0,...]
            fig, ax = plt.subplots(1,2)
            ax[0].scatter(dp[...,0]*par, -dvel[...,1])#, color = color2[0:num_features,...])
            ax[0].scatter(dp[...,1], dvel[...,0]*par)#, color = color2[0:num_features,...],marker = '*')
            # ax[1].hist(-dvel[...,1]/(dp[...,0]*par))
            # ax[1].scatter(dp[...,1], dvel[...,0]*par/(dp[...,1]))
            # ax[1].scatter(p0[...,0]*par, -vel[...,0])
            # ax[1].scatter(p0[...,1], vel[...,1]*par)
            
            ax[0].plot(dp[...,0]*par, -dp[...,0]*par*theta)
            ax[0].plot(dp[...,1], -dp[...,1]*theta)
            p1test = (p0*np.array((par, 1)))[:,0,:]@matrix[..., 0:2] + matrix[..., 2]
            ax[1].scatter(p1[...,0]*par, p1test[...,0])
            ax[1].scatter(p1[...,1], p1test[...,1])
            ax[1].plot([300,900], [300,900])
            ax[0].set_title(f'tetha = {theta}')
            ax[1].set_title(f'inliers = {inliers.sum()/len(inliers)}')
            plt.show()
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            these_colors = color.copy()
            these_colors[inliers] = np.array([255,255,255])
            lines = cv.line(lines, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv.circle(frame, (int(a), int(b)), 5, these_colors[i].tolist(), -1)
        img = cv.add(frame, lines)
        cv.imshow('frame', img)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
    cv.destroyAllWindows()
    
    return thetas, translation, 1/frame_rate* np.array(range(1,len(thetas)+1))


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
        fname = os.path.basename(video_file).split('/')[-1]
        wafer,chip,dev,delaytime, voltage = fname.split('_')
        voltage = voltage.split('V.')[0]
        thetas, translation, time = process_video_sparse(video_file)
        fig, ax = plt.subplots()
        ax.plot(time, 180/np.pi * np.cumsum(thetas))
        ax.scatter(time, 180/np.pi * np.cumsum(thetas))
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angular displacement (degrees)')
        fig.savefig(video_file + '.png')
        ax.set_title(video_file)
        data = {f'Time {voltage}V {delaytime} ms': time,
                f'theta {voltage}V {delaytime} ms':180/np.pi * np.cumsum(thetas),
                 f'translation x {voltage}V {delaytime} ms': translation[:,0],
                   f'translation y {voltage}V {delaytime} ms': translation[:,1]  }
        try:
            df = pd.read_csv(os.path.join(directory,f'{wafer}_{chip}_{dev}.csv'))
        except FileNotFoundError:
            df = pd.DataFrame()
        for key,value in data.items():
            df[key] = value
        # df_new = pd.concat([df,df2], axis = 1)
        df.to_csv(os.path.join(directory,f'{wafer}_{chip}_{dev}.csv'))
        # flow, _ = dense_flow.dense_flow_on_video(video_file)
        # basename_without_ext = os.path.splitext(os.path.basename(video_file))[0]
        # write_fname = os.path.join(directory, basename_without_ext+'.npy')
        # with open(write_fname, "wb") as f:
        #     np.save(f, flow.astype(np.float16))
# %%
if __name__ == '__main__':
    main()
# %%
