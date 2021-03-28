import numpy as np

import torch
from torch import optim

def get_optimizer(model):
    """ Here you choose your optimizer and define the hyperparameter space you
    would like to search.
    """
    opt = optim.SGD
    lr = np.random.choice(np.logspace(-5, 0, base=10))
    momentum = np.random.choice(np.linspace(0.1, .9999))
    return opt(model.parameters(), lr=lr, momentum=momentum)


def exploit_and_explore(top_chkpt_path, bot_chkpt_path, params,
                        perturb_factors=(1.2, 0.8)):
    """ Copy params from the better model and running avgs from the
    corresponding optimizer.
    """
    # Copy model parameters.
    chkpt = torch.load(top_chkpt_path)
    model_state = chkpt['model']
    opt_state   = chkpt['opt']
    batch_size  = chkpt['batch_size']

    for param_name in params['optimizer']:
        perturb = np.random.choice(perturb_factors)
        for param_group in opt_state['param_groups']:
            param_group[param_name] *= perturb

    if params['batch_size']:
        perturb = np.random.choice(perturb_factors)
        batch_size = int(np.ceil(perturb * batch_size))

    chkpt = dict(
        model=model_state,
        opt=opt_state,
        batch_size=batch_size
        )
    torch.save(chkpt, bot_chkpt_path)

