import numpy as np
import os
import json
import math
import random, os
import shutil

# Some simple code in order to copy and move data subset of complete dataset


def extract_data(class_path, class_path_move, nbr_files):
    # extract nbr_files amount of audio clips from class and move into new directory
    nbr = 1
    print('no')
    for file in os.listdir(class_path):
        if nbr == nbr_files:
            break

        print(nbr)
        source = class_path + '/'+ file
        print(source)
        dest = shutil.copy(source, class_path_move)
        nbr = nbr + 1 


# For extracting, destination and source folders
CLASS_PATH = "../Database_full/Airport" 
CLASS_PATH_MOVE = "../Database_500/Airport_reduced"

n = 500 # number of audiofiles to extract from full datatset

extract_data(CLASS_PATH, CLASS_PATH_MOVE, n)