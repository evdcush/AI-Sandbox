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

#==== Crossover
crossover : (genome1, genome2) ---> (genomeA, genomeB)
    reproduction in this GA is defined as a random,
    multi-point crossover between the weights of
    two different genomes, followed by mutation on the
    offspring of the two. Nothing fancy, just the
    selection of weights with masking algebra

# Evolving the genetic pool
#--------------------------
The process of evolution for this GA is as follows:

- Population of genomes is initialized

- Population is "evolved" through a fixed number of epochs,
  following the standard GA process:
  * Selection
    * Fitness
  * Crossover
    * Mutation
  * Population <--- Offspring of fittest genomes

- Using the (hopefully) fit or adapted population,
  predictions on the test dataset are made by
  taking the median label predictions across all genomes

"""
import os
import sys

import numpy as np

# gross hackky pathing for simple imports
cwd = str(os.path.abspath(os.path.dirname(__file__)))
dpath = cwd.replace('/nature/evolutionary', '/data')
sys.path.append(dpath)

from dataset import IrisDataset



# GA constants
#---------------------------------

POPULATION_SIZE = 128
TOURNAMENT_SIZE = 78
MUTATION_RATE = 0.05



def init_genome(size=(3,4)): # glorot normal
    scale = np.sqrt(2 / sum(size))
    return np.random.normal(scale=scale, size=size).astype(np.float16)

def init_pop():
    population = [init_genome() for _ in range(POPULATION_SIZE)]
    return population

def fitness(x, y, g):
    h = np.matmul(x, g.T) # (N, D).(D, 3)
    yhat = np.argmax(h, axis=1)
    return (y == yhat).sum()

def selection(x,y,pop):
    idx = np.random.choice(POPULATION_SIZE, TOURNAMENT_SIZE, replace=False)
    tournament = list(np.array(pop)[idx])
    fitnesses = [fitness(x,y,g) for g in tournament]
    fittest = tournament[np.argmax(fitnesses)]
    return fittest

def crossover(p1, p2):
    gene_mask = np.random.randint(0, 2, p1.shape, dtype=np.bool)
    c1 = (p1 * gene_mask) + (p2 * ~gene_mask)
    c2 = (p1 * ~gene_mask) + (p2 * gene_mask)
    return c1.astype(np.float16), c2.astype(np.float16)

def mutate(g):
    if np.random.random() < MUTATION_RATE:
        mut_seq = init_genome((1,4)).squeeze()
        seq_idx = np.random.choice(3)
        g[seq_idx, :] = mut_seq
    return g


#===============================================================
# Evolving a population


B = 12 # batch-size
pop = init_pop()
num_gens = 200
for step in range(num_gens):
    x, t = utils.IrisDataset.get_batch(np.copy(X), step, B)
    next_gen = []
    for i in range(POPULATION_SIZE // 2):
        p1 = selection(x,t,pop)
        p2 = selection(x,t,pop)
        child1, child2 = crossover(p1, p2)
        next_gen.append(mutate(child1))
        next_gen.append(mutate(child2))
    pop = next_gen



#===============================================================
# Evaluating population "fitness"
#  (eg, making predictions on a test set)

def predict(x, g):
    h = np.matmul(np.copy(x), g.T) # (N, 3)
    yhat = np.argmax(h, axis=1) # (N,)
    return yhat

def query_gene_pool(test_set, pool):
    yhats = np.zeros((len(pool), test_set.shape[0])).astype(np.int32)
    #x, t = utils.IrisDataset(np.copy(Y), 1, test=True)
    x, t = test_set[...,:-1], test_set[...,-1]
    t = t.astype(np.int32)
    for gidx, gene in enumerate(pool):
        yhats[gidx] = predict(x, gene)
    return yhats


test_preds = query_gene_pool(Y, pop)
Ypred = np.median(test_preds, axis=0).astype(np.int32)
accuracy = np.sum(Ypred == Y[...,-1])

