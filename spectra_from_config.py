#!/usr/bin/env python
# coding: utf-8

import json
import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d

# GLOBAL PARAMETERS

shift_range = 50 # for now, we shift all peaks independently
variation_range = 0.05 # +/- of absolute height for each peak
kernel_range = (2, 5) # min and max for guassian kernel sizes

n_test = 9 # predefined variations, just here for clarity
n_val = 10
n_train = 50

rng = np.random.default_rng(2022)

def vary_peaks(position_list, height_list):
    # since we set a boundary parameter, positions should never exceed range
    # still applying clip to be sure
    new_positions = np.clip(np.array([
        rng.integers(f-shift_range, f+shift_range) for f in position_list
        ]), 0, 4999)
    new_heights = np.clip(np.array([
        rng.uniform(f-variation_range, f+variation_range) for f in height_list
        ]), 0, 2)
    return new_positions, new_heights

def add_noise(input_scan):
    scan = input_scan.copy() #avoiding pointer confusion
    gaus = 1/3 * np.clip(rng.normal(0, 1, scan.size), -3, 3)
    gaus = (gaus*.5)+.5
    noise_lvl = rng.uniform(.01, .02)
    scan += gaus * (noise_lvl*np.max(scan))
    return scan  

def main():   
    with open('dataset_configs/dataset500.json', 'r') as file:
        config = json.load(file)
    datapoints = config['datapoints']
    n_classes = config['classes']
    # initialize arrays to fill
    x_train = np.zeros([(n_train*n_classes), datapoints])
    x_val = np.zeros([(n_val*n_classes), datapoints])
    x_test = np.zeros([(n_test*n_classes), datapoints])
    y_train = np.zeros(n_train*n_classes)
    y_val = np.zeros(n_val*n_classes)
    y_test = np.zeros(n_test*n_classes)

    spectra = config['spectra']
    for i, phase in enumerate(tqdm(spectra.keys())):
        peak_positions = np.array(spectra[phase]['peak_positions'])
        peak_heights = np.array(spectra[phase]['peak_heights'])
        
        # train data
        for j in range(n_train):
            scan = np.zeros(datapoints)
            # apply shifts
            new_pos, new_hi = vary_peaks(peak_positions, peak_heights)
            scan[new_pos] = new_hi
            # convolve with gaussian kernel
            scan = gaussian_filter1d(scan, rng.uniform(*kernel_range), 
                                     mode='constant')
            # adding gaussian noise to training data
            scan = add_noise(scan)
            x_train[(i*n_train)+j] = scan
        y_train[i*n_train:(i+1)*n_train] = i
        
        # validation data
        for j in range(n_val):
            scan = np.zeros(datapoints)
            # apply shifts
            new_pos, new_hi = vary_peaks(peak_positions, peak_heights)
            scan[new_pos] = new_hi
            # convolve with gaussian kernel
            scan = gaussian_filter1d(scan, rng.uniform(*kernel_range), 
                                     mode='constant')
            x_val[(i*n_val)+j] = scan
        y_val[i*n_val:(i+1)*n_val] = i
        
        # test - use exact values + defined variations
        scan = np.zeros([n_test, datapoints])
        scan[0,peak_positions] = peak_heights # exact reference
        # two variants with exact heights and -25/+25 position shifts
        pos_left = np.clip(np.array(peak_positions)-25, 0, 4999)
        pos_right = np.clip(np.array(peak_positions)+25, 0, 4999)
        scan[1,pos_left] = peak_heights
        scan[2,pos_right] = peak_heights
        # three variants with -25/0/+25 shifts and 0.2 higher intensities
        new_heights = [f+.2 if f != 1. else f for f in peak_heights]
        scan[3,peak_positions] = new_heights
        scan[4,pos_left] = new_heights
        scan[5,pos_right] = new_heights
        # three variants with -25/0/+25 shifts and 0.2 lower intensities
        new_heights = [f-.2 if f != 1. else f for f in peak_heights]
        scan[6,peak_positions] = new_heights
        scan[7,pos_left] = new_heights
        scan[8,pos_right] = new_heights
        for j in range(n_test):
            x_test[i*n_test+j] = gaussian_filter1d(scan[j], 2, mode='constant')
        y_test[i*n_test:(i+1)*n_test] = i   

    np.save('x_train.npy', x_train)
    np.save('y_train.npy', y_train)
    np.save('x_val.npy', x_val)
    np.save('y_val.npy', y_val)
    np.save('x_test.npy', x_test)
    np.save('y_test.npy', y_test)

if __name__ == '__main__':
    main()
