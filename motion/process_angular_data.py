# The point of this script is to read csv data with angle and time and compute different values, plot them, and output csv for desired plots
# %% 
import cv2 as cv
import numpy as np
import tkinter as tk
from tkinter import filedialog
from math import sqrt, atan2, degrees
import csv
import matplotlib.pyplot as plt
from IPython.display import Image, display
import background_remove
import os
import pandas as pd
# %% 

csv_file = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv")])

# %% 
directory = os.path.dirname(csv_file)
fname = os.path.basename(csv).split('/')[-1]