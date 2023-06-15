## this file is used to run the dense optical flow on all selected videos, and save csv files with the results to same directory
# %%
import numpy as np
import tkinter as tk
from tkinter import filedialog
import dense_flow
import os 
def main():
    root = tk.Tk()
    root.withdraw()

    video_files = filedialog.askopenfilenames(filetypes=[("MP4 files", "*.mp4")])
    if not video_files:
        print("No video file selected.")
        return
    for video_file in video_files:
        directory = os.path.dirname(video_file)
        flow, _ = dense_flow.dense_flow_on_video(video_file)
        basename_without_ext = os.path.splitext(os.path.basename(video_file))[0]
        write_fname = os.path.join(directory, basename_without_ext+'.npy')
        with open(write_fname, "wb") as f:
            np.save(f, flow.astype(np.float16))
if __name__ == '__main__':
    main()
# %%
