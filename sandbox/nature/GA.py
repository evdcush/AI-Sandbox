""" Simple Genetic Algorithm

NOTE: Current implementation defined only on the Iris dataset,
as a control/comparison to the deep model.


# The genetic algorithm is defined as follows:
#---------------------------------------------

#==== Genome
genome : ndarray.float32, (C, D)
    an array used to make predictions (affine trans.)
    on the iris class, from the observed input features
    where:
    C = |iris classes|  = 3
    D = |iris features| = 4

#==== Fitness function
fitness : (features, labels, genome) --> int
    fitness of a genome is defined as the number of
    correct predictions of class from a set of input
    features

#==== Selection routine
selection : (samples, population) ---> fittest genome
    tournament-style selection routine where
    TOURNAMENT_SIZE number of genomes are randomly
    selected and the fittest genome (singular) is returned

#==== Mutation
mutate : (genome,) ---> genome | mutated_genome
    with probability MUTATION_PROB, a genome has
    one of it's 3 weight vectors re-initialized

#==== Reproduction
reproduce : (genome1, genome2) ---> (genomeA, genomeB)
    reproduction in this GA is defined as a random,
    multi-point crossover between the weights of
    two different genomes, followed by mutation on the
    offspring of the two. Nothing fancy, just the
    selection of weights with masking algebra

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

import numpy as np

sys.path.insert(1, '..')
from data.dataset import IrisDataset



# GA constants
#---------------------------------

POPULATION_SIZE = 128
TOURNAMENT_SIZE = 78
MUTATION_RATE = 0.05
_dtype = np.float16

def init_genome(size=(3,4)):
    """ Genomes are:
         - (3,4) ndarray
         - sampled from Glorot random normal distribution
         - with low FP precision (to keep representation simple)
     # NOTE: not sure this is the sensical distibution to sample from,
             nor even whether the genomes should be sampled. Consideration
             for another time.
    """
    scale = np.sqrt(2 / sum(size))
    return np.random.normal(scale=scale, size=size).astype(_dtype)


def init_population():
    # initially diverse, the population should become
    # highly specialized over generations
    population = [init_genome() for _ in range(POPULATION_SIZE)]
    #return np.array(population) # (POPULATION_SIZE, 3, 4)
    return population

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def fitness_mat(x, y, sel):
    """ Essentially accuracy objective function

    Evaluates fitness over entire selection (ndarray) of genomes

    NB: The last step is non-differentiable, which is
        not a constraint on a GA as it would be with
        gradient-based policy

    Params
    ------
    x : ndarray.float16, (N, D)
        input features
    y : ndarray.int32, (N,)
        ground truth labels (iris classes) for input x
    sel : ndarray.float16, (S, 3, D)
        randomly selected genomes from population

    Returns
    -------
    score : ndarray.int, (S,)
        number of correct class predictions made by genome
    """
    #h = np.matmul(x, sel.T)
    h = np.einsum('nd,dks->nks', np.copy(x), sel.T)
    yhat = np.argmax(h, axis=1) # (N,S)
    scores = np.sum(yhat[:, None] == yhat, axis=0)
    return scores


def fitness(x, y, g):
    """ Essentially accuracy objective function.
    Evaluates how well adapted is the genome to its env?

    NB: The last step is non-differentiable, which is
        not a constraint on a GA as it would be with
        gradient-based policy

    Params
    ------
    x : ndarray.float16, (N, D)
        input features
    y : ndarray.int32, (N,)
        ground truth labels (iris classes) for input x
    g : ndarray.float16, (3, D)
        'genome' represention

    Returns
    -------
    score : int
        number of correct class predictions made by genome
    """
    h     = np.matmul(x, g.T)    # (N, D).(D, 3) ---> (N, 3)
    yhat  = np.argmax(h, axis=1) # (N,)
    score = (y == yhat).sum()
    return score

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def selection(x, y, population):
    """ Tournament style selection routine

    A fixed number of genomes are selected from the population at random
    and the fittest genome from that selection goes on to reproduce

    """
    idx = np.random.choice(POPULATION_SIZE, TOURNAMENT_SIZE, replace=False)
    tournament = list(np.array(population)[idx])
    fitnesses = [fitness(x,y,g) for g in tournament]
    fittest = tournament[np.argmax(fitnesses)]
    return fittest

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def reproduce(p1, p2):
    """ Crossover routine for two genomes

    Instead of more typical single-point transfer, a masking
    array is used to select multi-point transfer from
    parent genomes to child genomes
    """
    gene_mask = np.random.randint(0, 2, p1.shape, dtype=np.bool)
    #==== offspring from parent genomes
    c1 = (p1 *  gene_mask) + (p2 * ~gene_mask)
    c2 = (p1 * ~gene_mask) + (p2 *  gene_mask)
    return c1.astype(_dtype), c2.astype(_dtype)

def mutate(g):
    """ Mutation defined here as randomly replacing a certain
    part of the genome with a new sample
    """
    if np.random.random() < MUTATION_RATE:
        mut_seq = init_genome((1,4)).squeeze()
        seq_idx = np.random.choice(3)
        #g[seq_idx, :] = mut_seq
        g[seq_idx] = mut_seq
    return g


#===============================================================
# Evolving a population

# Setup
#---------
dataset = IrisDataset()
B = 12 # batch-size
population = init_population()
num_gens = 50

for step in range(num_gens):
    x, t = dataset.get_batch(step, B)
    next_gen = []
    for i in range(POPULATION_SIZE // 2):
        #==== selection
        p1 = selection(x, t, population)
        p2 = selection(x, t, population)
        #==== crossover
        child1, child2 = reproduce(p1, p2)
        next_gen.extend([mutate(child1), mutate(child2)])  # this sounds bad, change var naem
    population = next_gen



#===============================================================
# Evaluating population "fitness"
#  (eg, making predictions on a test set)

def predict(x, g):
    h = np.matmul(np.copy(x), g.T) # (N, 3)
    yhat = np.argmax(h, axis=1) # (N,)
    return yhat

def predict_mat(x, genomes):
    # x.shape = (30, 4)
    # g.shape = (P, 3, 4)
    h = np.einsum('nd,dkp->nkp', np.copy(x), genomes.T) # (N,3,POPULATION_SIZE)
    yhats = np.argmax(h, axis=1) # (N,POPULATION_SIZE)
    return yhats

def query_gene_pool(x, pool):
    yhats = np.zeros((len(pool), x.shape[0])).astype(np.int32)
    for gidx, gene in enumerate(pool):
        yhats[gidx] = predict(x, gene)
    return yhats

#==== split test set
X_test = dataset.X_test
x_test, y_test = X_test[...,:-1], X_test[...,-1]

test_preds = query_gene_pool(x_test, population) #(num_test, POPULATION_SIZE)

Y_pred   = np.median(test_preds, axis=0).astype(np.int32)
accuracy = np.sum(Y_pred == y_test) / len(y_test)

print(f'Y_pred: {Y_pred}')
print(f'Accuracy: {accuracy}')
