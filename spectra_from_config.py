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
datapoints = 5000

n_test = 3
n_val = 5
n_train = 10

def main():   
    with open('dataset_configs/dataset1.json', 'r') as file:
        config = json.load(file)

    n_phases = len(config.keys())

    x_train = np.zeros([(n_train*n_phases), datapoints])
    x_val = np.zeros([(n_val*n_phases), datapoints])
    x_test = np.zeros([(n_test*n_phases), datapoints])
    y_train = np.zeros(n_train*n_phases)
    y_val = np.zeros(n_val*n_phases)
    y_test = np.zeros(n_test*n_phases)

    rng = np.random.default_rng(2022)

    for i, phase in enumerate(tqdm(config.keys())):
        peaks = np.array(config[phase]['peak_positions'])
        peak_heights = np.array(config[phase]['peak_heights'])
        # test - use exact values + defined variations
        extra = 100
        scan = np.zeros(datapoints+2*extra)
        scan[peaks+extra] = peak_heights
        scan = gaussian_filter1d(scan, 2)
        x_test[i*n_test] = scan[extra:datapoints+extra]
        right_bound = extra+int(shift_range/2)
        left_bound = extra-int(shift_range/2)
        x_test[i*n_test+1] = scan[right_bound:datapoints+right_bound]
        x_test[i*n_test+2] = scan[left_bound:datapoints+left_bound]
        y_test[i*n_test:(i+1)*n_test] = i
        for j in range(n_val):
            # apply shift and clip peak positions outside range
            new_peaks = np.clip(np.array([rng.integers(f-shift_range, f+shift_range) for f in peaks]), 0, 4999)
            # apply height variation and clip peaks smaller 0
            new_heights = np.clip(np.array([rng.uniform(f-variation_range, f+variation_range) for f in peak_heights]), 0, 2)
            scan = np.zeros(datapoints)
            scan[new_peaks] = new_heights
            scan = gaussian_filter1d(scan, rng.uniform(*kernel_range))
            x_val[(i*n_val)+j] = scan
        y_val[i*n_val:(i+1)*n_val] = i
        for j in range(n_train):
            # apply shift and clip peak positions outside range
            new_peaks = np.clip(np.array([rng.integers(f-shift_range, f+shift_range) for f in peaks]), 0, 4999)
            # apply height variation and clip peaks smaller 0
            new_heights = np.clip(np.array([rng.uniform(f-variation_range, f+variation_range) for f in peak_heights]), 0, 2)
            scan = np.zeros(datapoints)
            scan[new_peaks] = new_heights
            scan = gaussian_filter1d(scan, rng.uniform(*kernel_range))
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
