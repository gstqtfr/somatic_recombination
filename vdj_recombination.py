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
def loop_with_gene_and_antibody_selection(sz_of_pop=20,
                                 sz_of_genome=10,
                                 sz_of_clonal_pool=20,
                                 sz_of_alphabet=256,
                                 n_iterations=100,
                                 epsilon=1e-1,
                                 verbose=False):

    # our "target": we need to find this particular genome (*big* search space)
    antigen = np.full(shape=(sz_of_genome,), fill_value=0, dtype='int16')
    # this is the maximum distance, as far away as possible from our target as
    # we can get
    maximum_distance = np.full(shape=(sz_of_genome), fill_value=(sz_of_alphabet-1), dtype='int16')

    beta = 1.0 / sz_of_alphabet

    # the repertoire is our population of "circulating" antibodies
    repertoire = recomb.initialize_population(sz_of_pop, sz_of_genome)
    # we give each gene a (fairly small) probability of being chosen
    repertoire_gene_affinity = recomb.initialize_gene_affinity(sz_of_alphabet, beta)

    # this is our loss function
    loss_function = partial(recomb.mean_absolute_error, antigen=antigen)

    # get the distance
    distances = recomb.get_distance(repertoire, loss_function=loss_function)

    # get the affinity of the gene expressed in each antibody
    repertoire_gene_affinity = recomb.gene_affinity_per_antibody(repertoire,
                                                                 distances,
                                                                 repertoire_gene_affinity,
                                                                 beta)

    for iteration in range(1, n_iterations):
        pass

    # dummy return statement, it'll do for now
    return repertoire, repertoire_gene_affinity, distances


def create_clonal_pool(antibody, weights, n_clone_pool, mutate_op):
    '''Takes an antibody & creates a pool of clones. They are then subjected to
        a mutation process
        :param antibody: numpy array representing the antibody
        :param weights: array of probabilities affecting selection of genes
        :param n_clone_pool: number of clones we're going to build
        :param mutate_op: partial function of somatic hypermutation operator
        :return: the hypermutated pool of clones
        '''
    # we don't want to change the antibody in the repertoire yet ...
    _antibody = np.copy(antibody)
    clonal_pool = np.tile(_antibody, (n_clone_pool, 1))
    for i in range(0, n_clone_pool):
        # mutate the flip out of it
        clonal_pool[i] = mutate_op(clonal_pool[i], weights)
    return clonal_pool
