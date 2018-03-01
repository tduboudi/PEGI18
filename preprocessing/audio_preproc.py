#!/usr/bin/python3.6

import re
import json
import sys
import glob
import os
import numpy as np

# First, we get the number of files in the given directory
nb_files = 0

directory = sys.argv[1]
os.chdir(directory)
for file in glob.glob("*.csv"):
    nb_files = nb_files + 1

film_name = output = sys.argv[2]
output = sys.argv[3]

data = np.array([])
# Then, we read each file in the directory, and add the audio features to the data array.
for i in range(nb_files-1):
    file = open(film_name+"_"+str(i).zfill(5)+".csv", "r")
    for line in file:
        if ((line != "") & (line[0] != '@')):
            print(line)
            current_features = line.split(',')
            data = np.append(data,current_features)
    file.close()

with open(output, 'w') as output_file:
    data.tofile(output_file,',')
