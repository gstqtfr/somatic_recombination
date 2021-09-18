#!/usr/env/python

from collections import defaultdict
from functools import partial

import numpy as np

import recombination_utils as recomb

# TODO: jolly good code! finish!
def iteration(repertoire, antigen, maximum_distance, affinities,
              sz_of_alphabet, weights, mutate_op, loss_function,
               beta, sz_of_genome, epsilon=1e-1, sz_of_clonal_pool=20):
    '''Go through the repertoire of antibodies, get their distance, clone them,
        then hypermutate them. Replace antibodies if a clone has hiher distance
        than the antibody in the repertoire.
        :param repertoire: array of antibodies
        :param antigen: the target sequence
        :param maximum_distance: the furthest away from the target, or the abs. size of the interval
        :param sz_of_alphabet: how large the "symbol set" is
        :param affinities: how close we are to the target
        :param weights: affinity of genes expressed in repertoire
        :param mutate_op: somatic hypermutation operator
        :param beta: default value for genes not currently expressed (dunno if we need this ...)
        :param sz_of_genome: length of the antibody array
        :param sz_of_clonal_pool: how many clones we're going to create
        :return: affinities, repertoire, weights'''

    # grim reaper process
    # get the antibody with the worst performance
    worst_antibody = (list(affinities.items())[-1])[0]
    repertoire[worst_antibody] = np.array([recomb.random_sequence(size=sz_of_genome)], dtype='int16')

    # our selection operator; we create a clonal pool, perform somatic hypermutation, then
    # replace the antibody in the repertoire, if appropriate
    for idx, current_distance in affinities.items():
        # first we create our clonal pool; copy the antibody & mutate the clones
        clonal_pool = create_clonal_pool(
            antibody=repertoire[idx],
            weights=weights,
            sz_of_genome=sz_of_genome,
            sz_of_clonal_pool=sz_of_clonal_pool,
            mutate_op=mutate_op)

        # let's get the affinity of the clone pool
        clone_affinities = get_repertoire_affinity(repertoire=clonal_pool,
                                                   antigen=antigen,
                                                   maximum_distance=maximum_distance,
                                                   max_gene_affinity=(sz_of_alphabet - 1))


        # let's find the clone with the highest affinity
        clone_distance = (list(clone_affinities.items())[0])[1]

        # see how it compares with the current antibody's distance
        if clone_distance >= current_distance:
            clone_idx = (list(clone_affinities.items())[0])[0]
            repertoire[idx] = clonal_pool[clone_idx]

    # TODO: move this to the bottom of the function? Or do this first?
    # get the distance of the antibodies
    #affinities = recomb.get_affinities(repertoire, loss_function=loss_function)

    affinities = get_repertoire_affinity(repertoire=repertoire,
                  antigen=antigen,
                  maximum_distance=maximum_distance,
                  max_gene_affinity=(sz_of_alphabet-1))

    weights = recomb.gene_affinity_per_antibody(repertoire,
                                                affinities,
                                                weights,
                                                sz_of_genome)

    weights = recomb.weights_normalisation(weights, beta, epsilon)

    return affinities, repertoire, weights

# FIXME: let's start, from the top, & build it up.
def loop_with_vdj_recombination(mutate_op,
                                loss_function,
                                sz_of_pop=20,
                                sz_of_genome=10,
                                sz_of_clonal_pool=20,
                                sz_of_alphabet=256,
                                n_iterations=100,
                                epsilon=1e-1):

    # JKK: this should all be the same ...
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

    # FIXME: we can stick this code below in a loop, once we have
    # FIXME: clonal pools & hypermutation

    affinities = get_repertoire_affinity(repertoire=repertoire,
                                         antigen=antigen,
                                         maximum_distance=maximum_distance,
                                         max_gene_affinity=(sz_of_alphabet-1))

    # get the affinity of the gene expressed in each antibody

    weights = recomb.gene_affinity_per_antibody(repertoire=repertoire,
                                                affinities=affinities,
                                                weights=weights,
                                                sz_of_genome=sz_of_genome)

    # now, we need to normalise the weights shonary
    weights = recomb.weights_normalisation(weights, beta, epsilon)

    distances = defaultdict(list)

    for n in range(1, n_iterations):
        affinities, repertoire, weights = iteration(loss_function=loss_function,
                                                    repertoire=repertoire,
                                                         antigen=antigen,
                                                         maximum_distance=maximum_distance,
                                                         affinities=affinities,
                                                         sz_of_alphabet=sz_of_alphabet,
                                                         weights=weights,
                                                         mutate_op=mutate_op,
                                                         beta=beta,
                                                         sz_of_genome=sz_of_genome)
        # early-stopping: are we close enough yet?
        # we need to find the distances (rather than the affinities ...)

        # then we need to see if we're within epsilon of the target



        avg_affinity = np.sum(np.array(list(affinities.values()))) / sz_of_pop
        print(f"{n}\t{avg_affinity}")

    return repertoire, affinities, weights



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

    # FIXME: JKK: this is fundamentally a bit fucked.
    # FIXME: JKK: this below gets the distance. lower is better.
    # get the distance
    distances = recomb.get_distance(repertoire, loss_function=loss_function)

    # get the affinity of the gene expressed in each antibody
    weights = recomb.gene_affinity_per_antibody(repertoire=repertoire,
                                                distances=distances,
                                                weights=weights,
                                                sz_of_genome=sz_of_genome)

    for n in range(1, n_iterations + 1):
        distances, repertoire, weights = iteration(
            repertoire=repertoire,
            affinities=distances,
            loss_function=loss_function,
            weights=weights,
            mutate_op=mutate_op,
            beta=beta,
            sz_of_genome=sz_of_genome,
            sz_of_clonal_pool=sz_of_clonal_pool)
        # early_stopping(affinities, epsilon=1e2)
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

def get_repertoire_affinity(repertoire,
                            antigen,
                            maximum_distance,
                            max_gene_affinity=255):
    '''Get the affinity of the repertoire of antibodies with the
        target, which is given by the antigen array.'''
    repertoire_affinity = defaultdict(list)

    for idx in range(0, repertoire.shape[0]):
        repertoire_affinity[idx] = get_antibody_affinity(
            antibody=repertoire[idx],
            antigen=antigen,
            maximum_distance=maximum_distance)

    # we return the affinities sorted in reverse order
    return dict(sorted(repertoire_affinity.items(),
                       key=lambda item: item[1],
                       reverse=True))


def get_antibody_affinity(antibody, antigen, maximum_distance, max_gene_affinity=255):
    '''Gets a normalised affinity from the antigen in the interval [0,1].
        The higher the affinity, the better.'''
    return np.sum(maximum_distance - np.abs(antibody - antigen)) / (antigen.shape[0] * max_gene_affinity)