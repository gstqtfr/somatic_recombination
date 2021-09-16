#!/usr/bin/env python

from functools import partial

import recombination_utils as recomb
import vdj_recombination as vdj


# repertoire, weights, distances = vdj.loop_with_gene_and_antibody_selection(
#         mutate_op=mutate_op,
#         sz_of_pop=30,
#         sz_of_clonal_pool=30,
#         n_iterations=1000)

def driver_function(n_trials=100, n_iterations=1000, sz_of_alphabet=256):
    # need to set up a mutation operator
    mutate_op = partial(recomb.get_mutants, sz_of_alphabet=sz_of_alphabet)

    for i in range(1, n_trials + 1):
        stop, _, _, _ = vdj.loop_with_gene_and_antibody_selection(
            mutate_op=mutate_op,
            n_iterations=n_iterations)
        print(f"{i}\t{stop}")


if __name__ == '__main__':
    driver_function()
