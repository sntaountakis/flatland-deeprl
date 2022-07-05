import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os

def get_level(difficulty, seed):
    levels = [
        {
            "n_agents": 3,
            "n_actions": 5,
            "width": 25,
            "height": 25,
            "max_cities": 2,
            "max_rails_between_cities": 2,
            "max_rails_in_city": 3,
            "seed": seed
        },
        {
            "n_agents": 5 ,
            "n_actions": 5,
            "width": 25,
            "height": 25,
            "max_cities": 2,
            "max_rails_between_cities": 2,
            "max_rails_in_city": 3,
            "seed": seed
        },
        {
            "n_agents": 7,
            "n_actions": 5,
            "width": 30,
            "height": 30,
            "max_cities": 3,
            "max_rails_between_cities": 2,
            "max_rails_in_city": 3,
            "seed": seed
        }
    ]
    return levels[difficulty-1]

def log_results(params, levels):
    alg = params.alg+'_Pr' if params.priority else params.alg
    
    session_id = 'LR:'+str(params.lrate) + '_BS:' + str(params.batch_size) +'_RBS'+str(params.buffer_size)+'_'+str(params.learn_freq)+'_EP:'+params.epochs+'/'+params.sil_epochs+'_'+params.comment
    log_dir = os.path.join('logs', alg, levels[params.level - 1], str(params.seed), session_id)
    return SummaryWriter(log_dir=log_dir)

def get_checkpoint_path(params, levels):
    '''
    Set the path for the stored trained model
    '''
    alg = params.alg+'_Pr' if params.priority else params.alg
    cwd = os.getcwd()
    checkpoint_path = os.path.join(cwd, 'checkpoints', levels[params.level - 1], alg, str(params.seed))
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)
    
    return checkpoint_path