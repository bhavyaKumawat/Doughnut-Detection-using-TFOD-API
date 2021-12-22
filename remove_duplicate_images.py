# Delete duplicate images from a directory
# How to consume?
# python remove_duplicate_images.py --dir <dataset>


import hashlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dir", required=True, help="Directory Path")
args = vars(ap.parse_args())

# change dir
os.chdir(args["dir"])

files_list = os.listdir('.')

# initialize a list and a dictionary
duplicates=[]
hash_keys=dict()

for index, filename in enumerate(os.listdir('.')):
    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            filehash = hashlib.md5(f.read()).hexdigest()
        if filehash not in hash_keys:
            hash_keys[filehash]=index
        else:
            duplicates.append((index,hash_keys[filehash]))
#print(duplicates)

for index in duplicates:
    print("Removed {} from {}".format(files_list[index[0]], args["dir"] ))
    os.remove(files_list[index[0]])

# change dir to parent dir    
os.chdir(r'../')
