#!/usr/bin/env python

from functools import partial

import numpy as np

import recombination_utils as recomb
import vdj_recombination as vdj


def driver_function(n_trials=100, n_iterations=1000, sz_of_alphabet=256, sz_of_genome=10):
    # need to set up a mutation operator
    antigen = np.full(shape=(sz_of_genome,), fill_value=0, dtype='int16')
    mutate_op = partial(recomb.get_mutants, sz_of_alphabet=sz_of_alphabet)
    loss_function = partial(recomb.mean_absolute_error, antigen=antigen)

    for i in range(1, n_trials + 1):
        stop, repertoire, affinities, weights = vdj.loop_with_vdj_recombination(
            mutate_op=mutate_op,
            loss_function=loss_function,
            n_iterations=n_iterations)

        print(f"{i}\t{stop}")


if __name__ == '__main__':
    driver_function()
