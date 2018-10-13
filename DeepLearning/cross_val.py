"""
Cross validation for parameter search script
"""
import gc
import numpy as np

import utils
import layers
import functions as F
import optimizers as Opt

#==================================================
#                   CV Setup                      #
#==================================================
# Model config
#------------------
SEEDS = [3310, 99467, 27189, 77771]
optimizer   = Opt.Adam
objective   = [F.SoftmaxCrossEntropy]
activations = [F.Sigmoid, F.SeLU, layers.Swish]

# Training config
#------------------
NUM_ITERS = 1000
batch_size = 6

# Dataset
#-------------------
"""
80/20 train/test split gives us an even 5-fold for cross val
Per fold:
 * 24 test samples
 * 96 train samples
"""
_X_train = np.load(utils.IRIS_TRAIN)
split_size = .8 # already default in dataset

# Target variable: channels
#--------------------------------------------------------------
K_IN  = 4
K_OUT = 3
MAX_DEPTH = 3
MAX_SIZE = 700 # limit on sum kernels for any given channels sample
CHANNELS = list(np.load('./TESTING/GEN_700_CHANNELS.npy'))

# Helpers
#--------------------------------------------------------------
def generate_dataset():
    x_copy = np.copy(_X_train)
    split_seed = np.random.choice(15000)
    _dataset = utils.IrisDataset(x_copy, split_size, split_seed)
    return _dataset

def init_trainer(chan, act, dset, seed):
    return utils.Trainer(chan, optimizer, F.SoftmaxCrossEntropy,
                         act, dataset=dset, steps=NUM_ITERS,
                         batch_size=batch_size, rng_seed=seed)

# Loss tracker
#--------------------------------------------------------------
#cv_dims = (len(SEEDS), len(activations), len(CHANNELS), NUM_ITERS, 2)
#cv_test_dims = cv_dims[:-2] + (24, 2)
cv_dims = (len(CHANNELS), len(activations), len(SEEDS), NUM_ITERS, 2)
cv_test_dims = cv_dims[:-2] + (24, 2)
#==== collections
CV_TRAIN_LOSS = np.zeros(cv_dims,      np.float32)
CV_TEST_LOSS  = np.zeros(cv_test_dims, np.float32)


#==================================================
#                    CV Train                     #
#==================================================
#for idx_seed, seed in enumerate(SEEDS):
#    for idx_act, act in enumerate(activations):
#        dataset = generate_dataset()
#        for idx_chan, channels in enumerate(CHANNELS):

for idx_chan, channels in enumerate(CHANNELS):
    print(f'{channels}')
    for idx_act, act in enumerate(activations):
        for idx_seed, seed in enumerate(SEEDS):
            dataset = generate_dataset()
            # copy data for safety
            dataset.X_train = np.copy(dataset.X_train)
            dataset.X_test  = np.copy(dataset.X_test)

            # make trainer and train
            trainer = utils.Trainer(channels, optimizer, F.SoftmaxCrossEntropy,
                         act, dataset=dataset, steps=NUM_ITERS,
                         batch_size=batch_size, rng_seed=seed)
            trainer()

            # Store loss
            trainer.opt = None
            for layer in trainer.model.layers:
                layer = None
            trainer.model = None
            trainer.dataset = None
            trainer.activation = None
            lh_train, lh_test = trainer.get_loss_histories()
            trainer = None
            gc.collect()
            CV_TRAIN_LOSS[idx_chan, idx_act, idx_seed] = lh_train
            CV_TEST_LOSS[ idx_chan, idx_act, idx_seed] = lh_test
    print(f'\n\nCHANNELS[{idx_chan}]: {channels}, avgs per activation:')
    avg_train = np.copy(CV_TRAIN_LOSS)[idx_chan].mean(axis=(1,2))
    err, acc = avg_train[...,0], avg_train[...,1]
    print('    ERR: {:.4f} | {:.4f} | {:.4f}'.format(err[0], err[1], err[2]))
    print('    ACC: {:.4f} | {:.4f} | {:.4f}'.format(acc[0], acc[1], acc[2]))

# Save results
#--------------------------------------------------------------
np.save('CV_sgd2_train_loss', CV_TRAIN_LOSS) # WARNING: large file!
np.save('CV_sgd2_test_loss',  CV_TEST_LOSS)
