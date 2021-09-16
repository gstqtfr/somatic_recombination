#!/usr/bin/env python

import numpy as np
import somatic_recombination as som

def driver_function(n_trials=100, n_iterations=1000):
    # n_iterations: paramater
    for i in range(1, n_trials):
        # need to implement early stopping here
        stop, _, _ = som.loop_with_antibody_selection(n_iterations=1000,
                                                      verbose=False)
        print(f"{i}\t{stop}")

if __name__ == '__main__':
    driver_function()