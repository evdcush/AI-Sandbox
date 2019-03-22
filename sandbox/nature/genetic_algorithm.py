""" Classic genetic algorithm

# Population evolution
#---------------------
The process of evolution for this GA is as follows:

- Population of genomes is initialized

- Population is "evolved" through a fixed number of epochs,
  or generations following the standard GA process:

  * Selection(Population)  # Random selection of some percentage pop
  * Fitness(select)        # Evaluate fitness of selected genomes
  * Reproduction(fittest)  # Fittest genomes mate, have offspring
  * Mutation(offspring)    # Offspring have random chance of mutation
  * Population   <---------- Offspring of fittest to new population

- Using the (hopefully) fit or adapted population,
  predictions on the test dataset are made by
  taking the median label predictions across all genomes

"""
import os
import sys
import code
import traceback

import numpy as np
from scipy import stats

# rel path to utils for dataset
fpath = os.path.abspath(os.path.dirname(__file__))
sys.path.append('/'.join(fpath.split('/')[:-1]))
import utilities as utils

datasets = utils.DATASETS # interface to dataset loader


#-----------------------------------------------------------------------------#
#                                   Config                                    #
#-----------------------------------------------------------------------------#

# Conf vars
# =========
# sess vars
_seed  = 123
_dname = 'iris'
_num_test = 24
_batch_size = 4

# ga spec
_population_size = 128
_tournament_size = 36
_mutation_rate   = 0.1
_num_generations = 200


# Arg parser
# ==========
cli = utils.CLI

cli.add_argument('-d', '--dataset', type=str, default=_dname,
    choices=list(datasets.keys()), help='dataset for model')

cli.add_argument('-p', '--population_size', type=int, default=_population_size,
    metavar='P', help='number of genomes in population')

cli.add_argument('-t', '--tournament_size', type=int, default=_tournament_size,
    metavar='T', help='number of genomes per tournament')

cli.add_argument('-m', '--mutation_rate', type=float, default=_mutation_rate,
    metavar='M', help='probability a genome has some params re-initialized')

cli.add_argument('-g', '--num_generations', type=int, default=_num_generations,
    metavar='G', help='number of generations over which the population will evolve')

cli.add_argument('-r', '--rng_seed', type=int, default=_seed, metavar='R',
    help='random seed for initialization of population')

cli.add_argument('-n', '--num_test', type=int, default=_num_test, metavar='N',
    help='number of test samples')


#-----------------------------------------------------------------------------#
#                               initialization                                #
#-----------------------------------------------------------------------------#

def init_genome(shape, init=utils.glorot_uniform):
    """ interface to initialization func specified by `init` kwarg

    genomes are functionally similar to a weight var in a network layer
    that is to say, genomes are used in a linear transformation of data

    Params
    ------
    shape : tuple(int, int)
        shape should correspond exactly to (num_features, num_classes)
    init : function
        initializer function, from utilities
    """
    return init(shape)


def init_population(shape, population_size, seed=_seed):
    """ initialize population of genomes """
    np.random.seed(seed)
    population = [init_genome(shape) for _ in range(population_size)]
    return population


#-----------------------------------------------------------------------------#
#                             Selection                                       #
#-----------------------------------------------------------------------------#

def predict(x, g):
    """ predict class label given data input x and genome g """
    #code.interact(local=dict(globals(), **locals()))
    h = np.matmul(x, g) # (N, D).(D, K) ---> (N, K)
    yhat = np.argmax(h, axis=-1)
    return yhat

def fitness(x, y, g):
    """ Genetic fitness function (objective function for GAs)

    Evaluates how well adapted a genome is to its env by simply
    measuring predictive accuracy

    Params
    ------
    x : ndarray.float16, (N, D)
        input features
    y : ndarray.int32, (N,)
        ground truth labels for input x
    g : ndarray.float16, (3, D)
        'genome' represention

    Returns
    -------
    score : int
        number of correct class predictions made by genome
    """
    yhat = predict(x, g)
    score = (y == yhat).sum()
    return score

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def selection(x, y, population, tournament_size=_tournament_size):
    """ Tournament style selection routine

    A fixed number of genomes are selected from the population at random
    and the fittest genome from that selection goes on to reproduce.

    Params
    ------
    x : ndarray.float32; (N, D)
        input data

    y : ndarray.int32; (N,)
        class labels

    population : list(ndarray)
        population of genomes

    tournament_size : int
        how many genomes in tournament

    Returns
    -------
    fittest : ndarray
        fittest genome from tournament
    """
    # Select genomes randomly from pop
    idx = np.random.choice(len(population), tournament_size, replace=False)
    #####################################################################################################################################
    # FIX THIS. Keep pop in array or something, this is smelly
    tournament = list(np.array(population)[idx])

    # Evaluate fitness
    fitnesses = [fitness(x,y,g) for g in tournament]
    fittest = tournament[np.argmax(fitnesses)]
    return fittest


#-----------------------------------------------------------------------------#
#                            Crossover                                        #
#-----------------------------------------------------------------------------#

def reproduce(p1, p2):
    """ Crossover routine for two genomes

    Instead of more typical single-point transfer, a masking
    array is used to select multi-point transfer from
    parent genomes to child genomes
    """
    # Crossover points
    gene_mask = np.random.randint(0, 2, p1.shape, dtype=np.bool)

    # Offspring from parent genomes
    c1 = np.where(gene_mask, p1, p2)
    c2 = np.where(gene_mask, p2, p1)
    return c1.astype(p1.dtype), c2.astype(p1.dtype)  # NECESSARY?  ##########################################

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def mutate(g, mutation_rate):
    """ Mutation defined here as randomly resampling part of the genome """
    mutation = init_genome(g.shape)
    mutated_genes = np.random.rand(*g.shape) < mutation_rate
    g = np.where(mutated_genes, mutation, g)
    return g

#-----------------------------------------------------------------------------#
#                             Genetic algorithms                              #
#-----------------------------------------------------------------------------#

def genetic_algorithm(dataset, num_gens, pop_size, tourney_size, mute_rate,
                      batch_size=_batch_size, seed=_seed):
    """ fully specified genetic algorithm for classification problems

    returns a population of genomes evolved on dataset over num_gens

    One important design choice made here is running two different
    tournaments to select two parents. The more efficient choice would
    be to simply select the fittest 2 genomes from a single tournament,
    but there is greater genetic diversity by selecting from two tourn.
    """

    # Initialize genetic pool
    # =======================
    num_feat  = dataset.X.shape[-1]
    num_class = len(dataset.target_names)
    gene_size = (num_feat, num_class)
    population = init_population(gene_size, pop_size, seed)

    # Evolve population
    for gen in range(num_gens):
        x, y = dataset.get_batch(batch_size)
        next_generation = []

        # Selection & Reproduction
        for _ in range(pop_size // 2):
            # tournament
            parent_1 = selection(x, y, population, tourney_size)
            parent_2 = selection(x, y, population, tourney_size)

            # crossover & mutation
            child_1, child_2 = reproduce(parent_1, parent_2)
            next_generation.append(mutate(child_1, mute_rate))
            next_generation.append(mutate(child_2, mute_rate))

        # update population
        population = next_generation
    return population

def evaluate_population(dataset, population, test=False):
    """ evaluate population's fitness against a non-training dataset
    Each genome in the population makes a "prediction" on a testing sample.
    The overall population prediction is computed as the mode of the
    genes' predictions.
    """
    # Get correct data set
    if test:
        X = np.copy(dataset.x_test)
        Y = np.copy(dataset.y_test)
    else:
        X = np.copy(dataset.x_validation)
        Y = np.copy(dataset.y_validation)

    # Predict class labels
    G = len(population)
    N = len(Y)
    Y_hat_genes = np.zeros((G, N), np.int32) # Y_hat[i,j] == gene[i] pred on Y[j]

    for i, genome in enumerate(population):
        Y_hat_genes[i] = predict(np.copy(X), genome)

    # Summarize fitness
    Y_hat_pop = stats.mode(Y_hat_genes, axis=0)[0]
    pop_fitness = np.sum(Y_hat_pop == Y) / Y.size
    test_type = 'TEST' if test else 'VALIDATION'
    print(f'GA fitness, {test_type} accuracy: {pop_fitness:.4f}')
    return Y_hat_pop


def main():
    # parse args
    # ==========
    args = cli.parse_args()

    # sess
    seed  = args.rng_seed
    dname = args.dataset
    num_test = args.num_test

    # dataset init
    dataset = datasets[dname]()
    dataset.split_dataset(num_test=num_test)

    # ga conf
    population_size = args.population_size
    tournament_size = args.tournament_size
    mutation_rate   = args.mutation_rate
    num_generations = args.num_generations

    # Run GA
    # ======
    population = genetic_algorithm(dataset, num_generations, population_size,
                                   tournament_size, mutation_rate, seed=seed)
    preds = evaluate_population(dataset, population, test=True)

    return 0

if __name__ == '__main__':
    try:
        ret = main()
    except:
        traceback.print_exc()
    sys.exit(ret)
