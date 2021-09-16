#!/usr/env/python

from functools import partial

import numpy as np

import recombination_utils as recomb


# TODO: jolly good code! finish!
def _iteration(repertoire, distances, loss_function, repertoire_gene_affinity, mutate_op,
               beta, sz_of_genome, sz_of_clonal_pool=20):
    '''Go through the repertoire of antibodies, get their distance, clone them,
        then hypermutate them. Replace antibodies if a clone has hiher distance
        than the antibody in the repertoire.
        :param repertoire: array of antibodies
        :param distances: how close we are to the target
        :param loss_function: obvious
        :param repertoire_gene_affinity: affinity of genes expressed in repertoire
        :param mutate_op: somatic hypermutation operator
        :param beta: default value for genes not currently expressed (dunno if we need this ...)
        :param sz_of_genome: length of the antibody array
        :param sz_of_clonal_pool: how many clones we're going to create
        :return: distances, repertoire, repertoire_gene_affinity'''

    # grim reaper process
    # get the antibody with the worst performance
    worst_antibody = (list(distances.items())[-1])[0]
    repertoire[worst_antibody] = np.array([recomb.random_sequence(size=sz_of_genome)], dtype='int16')

    # our selection operator; we create a clonal pool, perform somatic hypermutation, then
    # replace the antibody in the repertoire, if appropriate
    for idx, current_distance in distances.items():
        # first we create our clonal pool; copy the antibody & mutate the clones
        clonal_pool = create_clonal_pool(
            antibody=repertoire[idx],
            weights=repertoire_gene_affinity,
            sz_of_genome=sz_of_genome,
            sz_of_clonal_pool=sz_of_clonal_pool,
            mutate_op=mutate_op)
        # get how close the clones are to the optima
        clone_distances = recomb.get_distance(clonal_pool, loss_function)
        # replace if higher distance in the clonal pool
        # what's the clone with higher distance?
        clone_distance = (list(clone_distances.items())[0])[1]
        clone_idx = (list(clone_distances.items())[0])[0]
        # compare with the distance of the current antibody
        if clone_distance <= current_distance:
            repertoire[idx] = clonal_pool[clone_idx]

    # TODO: move this to the bottom of the function? Or do this first?
    # get the distance of the antibodies
    distances = recomb.get_distance(repertoire, loss_function=loss_function)

    repertoire_gene_affinity = recomb.gene_affinity_per_antibody(repertoire,
                                                                 distances,
                                                                 repertoire_gene_affinity,
                                                                 beta)

    return distances, repertoire, repertoire_gene_affinity


def loop_with_gene_and_antibody_selection(mutate_op,
                                          sz_of_pop=20,
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
    maximum_distance = np.full(shape=(sz_of_genome), fill_value=(sz_of_alphabet - 1), dtype='int16')

    beta = 1.0 / sz_of_alphabet

    # the repertoire is our population of "circulating" antibodies
    repertoire = recomb.initialize_population(sz_of_pop, sz_of_genome)
    # we give each gene a (fairly small) probability of being chosen
    weights = recomb.initialize_gene_affinity(sz_of_alphabet, beta)

    # this is our loss function
    loss_function = partial(recomb.mean_absolute_error, antigen=antigen)

    # get the distance
    distances = recomb.get_distance(repertoire, loss_function=loss_function)

    # get the affinity of the gene expressed in each antibody
    weights = recomb.gene_affinity_per_antibody(repertoire,
                                                distances,
                                                weights,
                                                beta)

    for n in range(1, n_iterations + 1):
        distances, repertoire, weights = _iteration(
            repertoire=repertoire,
            distances=distances,
            loss_function=loss_function,
            repertoire_gene_affinity=weights,
            mutate_op=mutate_op,
            beta=beta,
            sz_of_genome=sz_of_genome,
            sz_of_clonal_pool=sz_of_clonal_pool)
        # early_stopping(distances, epsilon=1e2)
        done, index = recomb.early_stopping(distances, epsilon)
        if done:
            if verbose:
                print(f"Reached convergence at iteration {n} within tolerance {epsilon}")
            return n, repertoire, weights, distances
        avg_distance = np.sum(np.array(list(distances.values()))) / sz_of_pop

        if verbose:
            print(f"{n}\t{avg_distance:.2f}")

        # dummy return statement, it'll do for now
    return n, repertoire, weights, distances


def create_clonal_pool(antibody, weights, sz_of_genome, sz_of_clonal_pool, mutate_op):
    '''Takes an antibody & creates a pool of clones. They are then subjected to
        a mutation process
        :param antibody: numpy array representing the antibody
        :param weights: array of probabilities affecting selection of genes
        :param sz_of_genome: how large our antibody is
        :param sz_of_clonal_pool: number of clones we're going to build
        :param mutate_op: partial function of somatic hypermutation operator
        :return: the hypermutated pool of clones
        '''
    # we don't want to change the antibody in the repertoire yet ...
    _antibody = np.copy(antibody)
    clonal_pool = np.tile(_antibody, (sz_of_clonal_pool, 1))
    for i in range(0, sz_of_clonal_pool):
        # get a random hotspot & length
        (hotspot, length) = recomb.get_hotspot_and_region(sz=sz_of_genome)

        # mutate the flip out of it
        mutant = mutate_op(length=length, weights=weights)
        # print(f"mutant: {mutant}")
        # print(f"B/4: antibody: {antibody}")
        for idx in range(length):
            pos = (hotspot + idx) % sz_of_genome
            # print(f"hotspot: {hotspot}\tlength: {length}\tindex: {idx}\tposition: {pos}")
            clonal_pool[i][pos] = mutant[idx]
        # print(f"NOW: antibody: {clonal_pool[i]}")

    return clonal_pool
