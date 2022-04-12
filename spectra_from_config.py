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

n_test = 3 # predefined variations, just here for clarity
n_val = 10
n_train = 50

def main():   
    with open('dataset_configs/dataset500.json', 'r') as file:
        config = json.load(file)
    datapoints = config['datapoints']
    n_classes = config['classes']

    x_train = np.zeros([(n_train*n_classes), datapoints])
    x_val = np.zeros([(n_val*n_classes), datapoints])
    x_test = np.zeros([(n_test*n_classes), datapoints])
    y_train = np.zeros(n_train*n_classes)
    y_val = np.zeros(n_val*n_classes)
    y_test = np.zeros(n_test*n_classes)

    rng = np.random.default_rng(2022)

    spectra = config['spectra']
    for i, phase in enumerate(tqdm(spectra.keys())):
        peaks = np.array(spectra[phase]['peak_positions'])
        peak_heights = np.array(spectra[phase]['peak_heights'])
        # test - use exact values + defined variations
        scan = np.zeros([3, datapoints])
        scan[0,peaks] = peak_heights
        scan[1,np.clip(np.array(peaks)-25, 0, 4999)] = peak_heights
        scan[2,np.clip(np.array(peaks)+25, 0, 4999)] = peak_heights
        x_test[i*n_test] = gaussian_filter1d(scan[0], 2, mode='constant')
        x_test[i*n_test+1] = gaussian_filter1d(scan[1], 2, mode='constant')
        x_test[i*n_test+2] = gaussian_filter1d(scan[2], 2, mode='constant')
        y_test[i*n_test:(i+1)*n_test] = i
        for j in range(n_val):
            # apply shift and clip peak positions outside range
            new_peaks = np.clip(np.array([rng.integers(f-shift_range, f+shift_range) for f in peaks]), 0, 4999)
            # apply height variation and clip peaks smaller 0
            new_heights = np.clip(np.array([rng.uniform(f-variation_range, f+variation_range) for f in peak_heights]), 0, 2)
            scan = np.zeros(datapoints)
            scan[new_peaks] = new_heights
            scan = gaussian_filter1d(scan, rng.uniform(*kernel_range), mode='constant')
            x_val[(i*n_val)+j] = scan
        y_val[i*n_val:(i+1)*n_val] = i
        for j in range(n_train):
            # apply shift and clip peak positions outside range
            new_peaks = np.clip(np.array([rng.integers(f-shift_range, f+shift_range) for f in peaks]), 0, 4999)
            # apply height variation and clip peaks smaller 0
            new_heights = np.clip(np.array([rng.uniform(f-variation_range, f+variation_range) for f in peak_heights]), 0, 2)
            scan = np.zeros(datapoints)
            scan[new_peaks] = new_heights
            scan = gaussian_filter1d(scan, rng.uniform(*kernel_range), mode='constant')
            gaus = 1/3 * np.clip(rng.normal(0, 1, scan.size), -3, 3)
            gaus = (gaus*.5)+.5
            noise_lvl = rng.uniform(.01, .02)
            scan += gaus * (noise_lvl*np.max(scan))
            x_train[(i*n_train)+j] = scan
        y_train[i*n_train:(i+1)*n_train] = i

    np.save('x_train.npy', x_train)
    np.save('y_train.npy', y_train)
    np.save('x_val.npy', x_val)
    np.save('y_val.npy', y_val)
    np.save('x_test.npy', x_test)
    np.save('y_test.npy', y_test)

if __name__ == '__main__':
    main()
