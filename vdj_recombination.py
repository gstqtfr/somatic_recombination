#!/usr/env/python

from functools import partial

import numpy as np

import recombination_utils as recomb


# TODO: jolly good code! finish!
def _iteration(repertoire, loss_function, repertoire_gene_affinity, beta, sz_of_genome,
               n_bits=32, n_clonal_pool=20):
    '''Go through the repertoire of antibodies, get their distance, clone them,
        then hypermutate them. Replace antibodies if a clone has hiher distance
        than the antibody in the repertoire.'''

    # TODO: move this to the bottom of the function? Or do this first?
    # get the distance of the antibodies
    distances = recomb.get_distance(repertoire, loss_function=loss_function)

    repertoire_gene_affinity = recomb.gene_affinity_per_antibody(repertoire,
                                                                 distances,
                                                                 repertoire_gene_affinity,
                                                                 beta)
    # grim reaper process
    # get the antibody with the worst performance
    worst_antibody = (list(distances.items())[-1])[0]
    repertoire[worst_antibody] = np.array([recomb.random_sequence(size=sz_of_genome)], dtype='int16')

    # TODO: fill in code that, almost magically, makes this algorithm work

    return distances, repertoire, repertoire_gene_affinity


# FIXME: we'll need some params here ...
# def loop(popsz, size, clonal_pool_size, sz_of_alphabet, loss_function):
def loop_with_gene_and_antibody_selection():
    # how big our repertoire is
    popsz = 20
    # how many clones we create
    clonal_pool_sz = 30
    # this is the total number of potential genes
    sz_of_alphabet = 256
    # ... & this is how many genes build an antibody
    sz_of_genome = 5
    # number of times we loop around, cloning & mutating ...
    n_iterations = 100
    # our "target": we need to find this particular genome (*big* search space)
    antigen = np.full(shape=(sz_of_genome,), fill_value=0, dtype='int16')
    # this is the maximum distance, as far away as possible from our target as
    # we can get
    maximum_distance = np.full(shape=(sz_of_genome), fill_value=255, dtype='int16')

    beta = 1.0 / sz_of_alphabet

    # the repertoire is our population of "circulating" antibodies
    repertoire = recomb.initialize_population(popsz, sz_of_genome)
    # we give each gene a (fairly small) probability of being chosen
    repertoire_gene_affinity = recomb.initialize_gene_affinity(sz_of_alphabet, beta)

    # this is our loss function
    loss_function = partial(recomb.mean_absolute_error, antigen=antigen)

    # get the distance
    distances = recomb.get_distance(repertoire, loss_function=loss_function)

    repertoire_gene_affinity = recomb.gene_affinity_per_antibody(repertoire,
                                                                 distances,
                                                                 repertoire_gene_affinity,
                                                                 beta)

    for iteration in range(1, n_iterations):
        pass

    # dummy return statement, it'll do for now
    return repertoire, repertoire_gene_affinity, distances
