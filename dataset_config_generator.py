#!/usr/bin/env python
# coding: utf-8

"generate a dataset config (json) based on predefined parameters"

import json
import numpy as np

# PARAMETERS
n_datapoints = 5000
n_classes = 500
min_peaks = 2
max_peaks = 10
max_height = 100

def main():
    rng = np.random.default_rng(2022)
    config = {
        'datapoints' : n_datapoints,
        'classes' : n_classes,
        'min_peaks' : min_peaks,
        'max_peaks' : max_peaks,
        'max_height' : max_height
        }
    spectra = {}
    for phase in range(n_classes):
        n_peaks = rng.integers(min_peaks, max_peaks, endpoint=True)
        peak_positions = rng.integers(0, n_datapoints, n_peaks, endpoint=True)
        peak_heights = rng.integers(0, max_height, n_peaks)
        peak_heights = np.round(peak_heights / np.max(peak_heights), 3)
        phase_dict = {'peak_positions': np.sort(peak_positions).tolist(),
                      'peak_heights' : peak_heights[np.argsort(peak_positions)].tolist()}
        spectra[phase] = phase_dict
    
    config['spectra'] = spectra
    with open('dataset_configs/dataset500.json', 'w') as file:
        json.dump(config, file)

if __name__ == '__main__':
    main()
