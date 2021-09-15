import numpy as np

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
    """given the distances calc'd by the loss function, return the most distant antibody"""
    reversed_distance = dict(sorted(distance.items(),
                                    key=lambda item: item[1],
                                    reverse=True))
    return (list(reversed_distance.items())[0])[0]

def gene_affinity_per_antibody(repertoire, affinities, repertoire_gene_affinity, beta):
    """Gets the affinity (distance from) the target antigen for each antibody
        in the repertoire"""
    for idx in range(repertoire.shape[0]):
        # to make the code a little simpler to follow, we'll
        # create a reference to the antibody
        antibody = repertoire[idx]
        # get the affinity corresponding to this antibody
        antibody_affinity = affinities[idx]

        # now we iterate over each gene in the antibody ...
        for gene in antibody:
            repertoire_gene_affinity[gene] = antibody_affinity * beta

    # now we need to add beta tp any genes that aren't currently
    # expressed in the repertoire, so they have a probability of
    # being selected

    return repertoire_gene_affinity

# a couple of simple loss functions, used to find the affinity between
# the antibody and the antigen

def mean_absolute_error(antigen, antibody):
    return np.sum(np.abs(antigen - antibody)) / antigen.shape[0]


def root_mean_square_error(antigen, antibody):
    return np.sqrt(np.sum((antigen - antibody) ** 2.0) / antigen.shape[0])

