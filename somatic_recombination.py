#!/usr/env/python

import numpy as np

def random_sequence(size, n=255):
    return np.random.choice(n, size)

def initialize_population(n_pop, size):
    return np.array([random_sequence(size=size) for s in range(n_pop)])

def mean_absolute_error(antigen, antibody):
    return np.sum(np.abs(antigen - antibody)) / antigen.shape[0]

def root_mean_square_error(antigen, antibody):
    return np.sqrt(np.sum((antigen - antibody) ** 2.0) / antigen.shape[0])