#%%
from moviepy.editor import VideoFileClip
from tkinter import filedialog
import tkinter as tk
import os
def convert_avi_to_mp4(avi_file, mp4_file):
    # Load the AVI file
    if os.path.isfile(mp4_file):
        print(f'Skipping {avi_file}, mp4 already exists')
        return
    video = VideoFileClip(avi_file)
    
    # Convert to MP4
    video.write_videofile(mp4_file, codec='libx264', audio_codec='aac')
    
    # Close the video clip
    video.close()

# List of AVI files to convert
# root = tk.Tk()
# root.withdraw()
# avi_files = filedialog.askopenfilenames(filetypes=[("AVI files", "*.avi")])
# if not avi_files:
#     print("No video file selected.")
    
# for avi_file in avi_files:
#     mp4_file = avi_file.replace(".avi", ".mp4")
#     convert_avi_to_mp4(avi_file, mp4_file)
# %% convert all in folders
root = tk.Tk()
root.withdraw()
folder = filedialog.askdirectory()
avi_files = []
for root, dirs, files in os.walk(folder):
   for name in files:
      print(os.path.join(root, name))
      if 'avi' in name.split('.')[-1] and 'converted' not in name:
          avi_files.append(os.path.join(root,name))

# %%
mp4_files = []
for root, dirs, files in os.walk(folder):
   for name in files:
      print(os.path.join(root, name))
      if 'mp4' in name.split('.')[-1] and 'converted' not in root:
          mp4_files.append(os.path.join(root,name))
print(len(avi_files), len(mp4_files))

# %%
failed_conversions = []
for avi_file in avi_files:
    mp4_file = avi_file.replace(".avi", ".mp4")
    if mp4_file not in mp4_files:
        try:
            convert_avi_to_mp4(avi_file, mp4_file)
        except Exception as e:
            failed_conversions.append((avi_file, e))

# %%
mp4_files = []
for root, dirs, files in os.walk(folder):
   for name in files:
      print(os.path.join(root, name))
      if 'mp4' in name.split('.')[-1] and 'converted' not in root:
          mp4_files.append(os.path.join(root,name))
print(len(avi_files), len(mp4_files))
# %%
