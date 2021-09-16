#!/usr/env/python

from collections import defaultdict
from functools import partial
import numpy as np
import recombination_utils as recomb


def loop_with_antibody_selection(sz_of_pop=20,
                                 sz_of_genome=5,
                                 sz_of_clonal_pool=20,
                                 sz_of_alphabet=256,
                                 n_iterations=100,
                                 epsilon=1e-1,
                                 verbose=False):
    """
    :param sz_of_pop: how big our repertoire is
    :param sz_of_genome: how many genes our antibodies consist of
    :param sz_of_clonal_pool: how many clones we create
    :param sz_of_alphabet: this is the total number of potential genes
    :param n_iterations: number of times we loop around, cloning & mutating ...
    :param epsilon: tolerance - if within this distance to the optima, we stop adapting
    :param verbose: spit out more info
    :returns    current_iteration: stppping iteration
                repertoire: array of antibodies
                distances: how far each antibody is away from the optima
    """

    # our "target": we need to find this particular genome (*big* search space)
    antigen = np.full(shape=(sz_of_genome,), fill_value=0, dtype='int16')

    # now we need to construct a mutate operator; note, this could be passed
    # as a parameter if necessary
    mutate_op = partial(recomb.contiguous_somatic_hypermutation,
                        sz_of_genome=sz_of_genome,
                        sz_of_alphabet=sz_of_alphabet)

    # & we also need a loss function to get the antibody with the highest affinity
    loss_function = partial(recomb.mean_absolute_error, antigen=antigen)

    # randomly initialise our repertoire of antibodies
    repertoire = recomb.initialize_population(sz_of_pop, sz_of_genome)

    distances = defaultdict(list)
    current_iteration = 0

    for current_iteration in range(1, n_iterations + 1):
        repertoire, distances = iteration(repertoire, loss_function,
                                          mutate_op, sz_of_genome, sz_of_clonal_pool)
        done, index = recomb.early_stopping(distances, epsilon=epsilon)
        if done:
            if verbose:
                print(f"Reached convergence at iteration {current_iteration} within tolerance {epsilon}")
            return current_iteration, repertoire, distances
        if verbose:
            print(f"{distances}")

    return current_iteration, repertoire, distances


def iteration(repertoire, loss_function, mutate_op, sz_of_genome, sz_of_clonal_pool):
    '''Single-step iteration of the algorithm. Go through the repertoire of antibodies,
        get their distance, clone them, then hypermutate them. Replace antibodies if a
        clone has higher affinity than the antibody in the repertoire. Replace the worst
        performing antibody by randomly initialising it.
        :param repertoire: array of antibodies
        :param loss_function: function we're trying to minimise, like MAE or RMSE
        :param mutate_op: operator applied to the antibodies, like contiguous hypermutation
        :param sz_of_genome: how many genes our antibody is built from
        :param sz_of_clonal_pool: how many clones we're going to create
        :return: the modified repertoire and the latest repertoire distances to the optima  '''
    # see how our current antibody repertoire is doing ...
    distances = recomb.get_distance(repertoire, loss_function)
    # clone, hypermutate, & replace where appropriate
    repertoire = selection(distances, repertoire, loss_function, mutate_op, sz_of_clonal_pool)

    # grim reaper process
    # get the antibody with the worst performance
    worst_antibody = (list(distances.items())[-1])[0]
    # ... & replace it with a random antibody
    repertoire[worst_antibody] = np.array([recomb.random_sequence(size=sz_of_genome)], dtype='int16')

    return repertoire, distances


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


def get_gene_frequency(repertoire):
    '''Given the antibody repertoire, calculate how often a gene
        appears in the repertoire.'''
    (_unique, _counts) = np.unique(repertoire, return_counts=True)
    unique = np.array(_unique, 'int16')
    counts = np.array(_counts, 'int16')
    return np.array((unique, counts)).T


def get_gene_relative_frequency(gene_frequency, n_pop, antibody_size):
    '''Given a 2d array of the frequency of genes, of the form
        [[gene1, frequency1], [gene2, frequency2]] ..., calculate
        the relative frequency with which the gene occurs.'''
    relative_frequencies = defaultdict(list)
    # this is the maximum possible frequency - if all the antibodies were
    # built of the same gene (pretty boring repertoire)
    max_frequency = n_pop * antibody_size
    for gene, freq in gene_frequency:
        relative_frequencies[gene] = freq / max_frequency
    return relative_frequencies


def selection(distances, repertoire, loss_function, mutate_op, sz_of_clonal_pool):
    ''' Selection iterates over the antibody repertoire, creating a clonal
        pool for each antibody. The clones are then evaluated by the loss
        function, and replace the antibody if they have a higher distance.'''

    for idx, current_distance in distances.items():
        clonal_pool = create_clonal_pool(repertoire[idx],
                                         sz_of_clonal_pool,
                                         mutate_op=mutate_op)
        clone_distances = recomb.get_distance(clonal_pool, loss_function)

        # replace if higher distance in the clonal pool
        # what's theclone with higher distance?
        clone_distance = (list(clone_distances.items())[0])[1]
        clone_idx = (list(clone_distances.items())[0])[0]
        # compare with the distance of the current antibody
        if clone_distance <= current_distance:
            repertoire[idx] = clonal_pool[clone_idx]

    return repertoire


# this one's okay here, it's different  ...
def create_clonal_pool(antibody, n_clone_pool, mutate_op):
    '''Takes an antibody & creates a pool of clones. They are then subjected to
        a mutation process
        :param antibody: numpy array representing the antibody
        :param n_clone_pool: number of clones we're going to build
        :param mutate_op: partial function of somatic hypermutation operator
        :return: the hypermutated pool of clones'''
    # we don't want to change the antibody in the repertoire yet ...
    _antibody = np.copy(antibody)
    clonal_pool = np.tile(_antibody, (n_clone_pool, 1))
    for i in range(0, n_clone_pool):
        # mutate the flip out of it
        clonal_pool[i] = mutate_op(clonal_pool[i])
    return clonal_pool


# TODO: move ...
# this function turns our 1D antibody into a pair of
# real numbers, so that we can plug them into a function
def decode(interval, n_bits, bitstring):
    decoded = list()
    largest = 2 ** n_bits
    for i in range(len(interval)):
        # extract the substring
        start, end = i * n_bits, (i * n_bits) + n_bits
        substring = bitstring[start:end]
        # convert bitstring to a string of chars
        chars = ''.join([str(s) for s in substring])
        # convert string to integer
        integer = int(chars, 2)
        # scale integer to desired range
        value = interval[i][0] + (integer / largest) * (interval[i][1] - interval[i][0])
        # store
        decoded.append(value)
    return np.array(decoded)
