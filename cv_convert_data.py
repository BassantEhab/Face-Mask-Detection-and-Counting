# -*- coding: utf-8 -*-
"""cv_convert_data.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1iNhEjosaPt9FLW2qcPTN0GDxp_Uu3xFd
"""

# Importing the required libraries
import xml.etree.ElementTree as Xet
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
import cv2
from tqdm import tqdm
import os
import numpy as np
import csv
from random import shuffle 
from bs4 import BeautifulSoup
import pandas as pd

def Convert_files (dir):
  data = []
  for file in tqdm(os.listdir(dir)):
    path = os.path.join(dir,file)
    # Open XML file
    file = open(path, 'r')

    # Read the contents of that file
    contents = file.read()

    soup = BeautifulSoup(contents, 'xml')

    # Extracting the data
    filename = soup.find('filename')
    obj = soup.find_all('object')
    name = soup.find_all('name')
    xmin = soup.find_all('xmin')
    ymin = soup.find_all('ymin')
    xmax = soup.find_all('xmax')
    ymax = soup.find_all('ymax')

    # Loop to store the data in a list named 'data'
    for i in range(0, len(obj)):
      rows = [filename.get_text(), name[i].get_text(), xmin[i].get_text(), ymin[i].get_text(), xmax[i].get_text(), ymax[i].get_text()]
      data.append(rows)

  # Converting the list into dataframe
  df = pd.DataFrame(data, columns=["filename","name", "xmin", "ymin", "xmax", "ymax"], dtype = float)
  df = df.replace(r'\n',  '', regex=True)
  df = df.replace(r' ',  '', regex=True)

  display(df)
  
  # df.to_csv('/content/drive/MyDrive/CV project/data.csv')

Convert_files ('/content/drive/MyDrive/CV project/train/annotations')