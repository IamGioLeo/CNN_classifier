import os
import csv
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

header = ["K_SIZE", #KERNEL SIZE 
        "CONV_FIL", #THE AMOUNT OF CONVOLUTIONAL FILTERS AT EACH LAYER
        "LR", #LEARNING RATE
        "OPT", #CHOSEN OPTIMIZER
        "M", #MOMENTUM
        "WD",#WEIGHT DECAY
        "P", #PATIENCE
        "B_SIZE", #BATCH SIZE
        "ES", #TOTAL EPOCHS FOR THAT SESSION
        "E", #EPOCH OF THE RESULT
        "B_NORM", #IF BATCH NORMALIZATION IS APPLIED 
        "D_OUT", #CHOSEN DROPOUT PROBABILIT, IF MISSING NO DROPOUT
        "ENS", # IF THE RESULT REGADS AN ESNAMBLE OF NETWORKS
        "V_L", #RESULTED VAL LOSS
        "TR_L", #RESULTED TRAIN LOSS
        "V_ACC", #RESULTED VAL ACCURACY
        "TR_ACC", #RESULTED TRAIN ACCURACY
        "TE_ACC", #RESULTED TEST ACCURACY
        "DATA_AUG", #IF THE DATASET IS BASE, MIRRORED, OR AUGMENTED 
        "DATA_V", #VERSION OF THE DATA SPLIT
        ] 


def insert_in_csv(k_size=None, conv_fil=None, lr=None, opt=None, m=None, wd=None,
                  p=None, b_size=None, epochs=None, epoch=None, b_norm=None,
                  d_out=None, ens=None, v_l=None, tr_l=None,
                  v_acc=None, tr_acc=None, te_acc=None, data_aug=None,
                  data_v=None, csv_name="csv_v1.csv"):       
        
        row = [
        k_size, conv_fil, # I want it to stop if this are missing
        _nan_if_none(lr), _empty_if_none(opt),
        _nan_if_none(m), _nan_if_none(wd),
        _nan_if_none(p), _nan_if_none(b_size),
        _nan_if_none(epochs), _nan_if_none(epoch),
        _false_if_none(b_norm), _nan_if_none(d_out),
        _false_if_none(ens), _nan_if_none(v_l),
        _nan_if_none(tr_l), _nan_if_none(v_acc), 
        _nan_if_none(tr_acc), _nan_if_none(te_acc), 
        _empty_if_none(data_aug), _empty_if_none(data_v),
        ]
        
        file_exists = os.path.isfile(csv_name)
        write_header = not file_exists or os.path.getsize(csv_name) == 0

        with open(csv_name, mode="a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            writer.writerow(row)
        return


def _nan_if_none(x):
    return math.nan if x is None else x

def _false_if_none(x):
    return False if x is None else x

def _empty_if_none(x):
    return "" if x is None else x