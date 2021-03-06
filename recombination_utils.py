import numpy as np

def get_mutants(length, sz_of_alphabet, weights=None):
    """
    Takes a length of a region on the antibody, and the number of genes, and
    returns an array of mutant genes.Probability of selection of each gene
    being selected can be given in the weights parameter.
    :param length: the length of the contiguous region to mutate
    :param sz_of_alphabet: how many different elements make up our genes
    :param weights: if specified, the vector of probabilities used to select each gene
    :return: array of genes
    """
    if weights is None:
        return np.random.choice(sz_of_alphabet, length, p=weights)
    else:
        return np.random.choice(sz_of_alphabet, length)


def contiguous_somatic_hypermutation(antibody, sz_of_genome, sz_of_alphabet, weights=None):
    (hotspot, length) = get_hotspot_and_region(sz_of_genome)
    # print(f"hotspot: {hotspot} : length: {length}")
    if weights is None:
        mutants = get_mutants(length=length, sz_of_alphabet=sz_of_alphabet, weights=None)
    else:
        mutants = get_mutants(length=length, sz_of_alphabet=sz_of_alphabet, weights=weights)
    for idx in range(length):
        index = (hotspot + idx) % sz_of_genome
        antibody[index] = mutants[idx]
    return antibody

def get_hotspot_and_region(sz):
    '''Select a hotspot and contiguous region on the genome'''
    (hotspot, length) = np.random.choice(sz, 2)
    # We're going to mutate, so let's make sure we have *at least* one
    # hotspot
    if length == 0:
        length = 1
    return hotspot, length

def random_sequence(size, n=255):
    '''Return a random sequence, used to initialize antibodies or the
        initial repertoire of antibodies'''
    return np.random.choice(n, size)


# TODO: factor these guys into another module
def initialize_population(n_pop, size):
    '''Create a reportoire of randomly-initialized antibodies'''
    return np.array([random_sequence(size=size) for s in range(n_pop)], dtype='int16')


def initialize_gene_affinity(sz_of_alphabet, beta):
    '''Create our gene affinity vector. This will represent our probabilities of
        a gene being chosen.'''
    # we make sure that each gene has *some* initial value
    repertoire_gene_affinity = np.full(shape=(sz_of_alphabet), fill_value=beta)
    return repertoire_gene_affinity


# this one's okay here, different dynamics ...
def get_distance(antibodies, loss_function):
    """ get_distance calculates the loss of each antibody"""
    distance = {}
    for i in range(0, len(antibodies)):
        distance[i] = loss_function(antibody=antibodies[i])
    return dict(sorted(distance.items(), key=lambda item: item[1]))


def get_worst_distance(distance):
    """given the affinities calc'd by the loss function, return the most distant antibody"""
    reversed_distance = dict(sorted(distance.items(),
                                    key=lambda item: item[1],
                                    reverse=True))
    return (list(reversed_distance.items())[0])[0]


def weights_normalisation(raw_weights, beta=1/256, epsilon=1e-1):
    """Normalises the weights vector in the range [0,1]. Any weight which
        is assigned zero from this procedure gets assigned a small positive value
        :param raw_weights: the weights, in order of descending magnitude, of the genes
        :param beta: default, small, positive number for any weights set to zero
        :return: the normalised weights array"""
    max_weight = raw_weights[0]
    min_weight = raw_weights[-1]
    cooked_weights=np.zeros(shape=raw_weights.shape)
    if np.abs(max_weight - min_weight) <= epsilon:
        cooked_weights = (raw_weights - min_weight) / epsilon
    else:
        cooked_weights = (raw_weights - min_weight) / (max_weight - min_weight)
    # find any weights which have been set to 0
    zero_indices = np.where(cooked_weights == 0)[0]
    # ... & give them a small positive value, so there's still *some* probablity
    # of being selected
    cooked_weights[zero_indices] = beta

    return cooked_weights


def gene_affinity_per_antibody(repertoire, affinities, weights, sz_of_genome):
    """Gets the affinity (distance from) the target antigen for each antibody
        in the repertoire"""
    #print(f"Number of antibodies / shape of repertoire: {repertoire.shape[0]}")
    for idx in range(repertoire.shape[0]):
        # to make the code a little simpler to follow, we'll
        # create a reference to the antibody
        antibody = repertoire[idx]
        #print(f"Selected antibody repertoire[{idx}] is {antibody}")
        # get the affinity corresponding to this antibody
        antibody_affinity = affinities[idx]
        #print(f"Selected antibody repertoire[{idx}] has affinity {antibody_affinity}")
        # now we iterate over each gene in the antibody ...
        for gene in antibody:
            #weights[gene] = antibody_affinity * beta
            # FIXME: this'll have to be a little more sophisticated, but
            # FIXME: this'll do for now ...
            #print(f"gene is {gene}")
            #print(f"b/4: weights[{gene}] is {weights[gene]}")
            weights[gene] += antibody_affinity / sz_of_genome
            #print(f"NOW: weights[{gene}] is {weights[gene]}")
    #print()

    return weights


def early_stopping(distances, epsilon=1e2):
    '''
    early_stopping: If we are within a relatively small distance of the
    antigen, we stop adapting the antibodies.
    '''
    for idx, distance in distances.items():
        if distance <= epsilon:
            return True, idx
    return False, -1

# a couple of simple loss functions, used to find the affinity between
# the antibody and the antigen

def mean_absolute_error(antigen, antibody):
    return np.sum(np.abs(antigen - antibody)) / antigen.shape[0]


def root_mean_square_error(antigen, antibody):
    return np.sqrt(np.sum((antigen - antibody) ** 2.0) / antigen.shape[0])
