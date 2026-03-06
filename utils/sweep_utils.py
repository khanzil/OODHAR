import numpy as np
import copy

def get_random_search_configs(cfg, train_id):
    new_cfg = copy.deepcopy(cfg)
    new_cfg['train_id'] = f'no {train_id}'


    # same for every configs
    new_cfg['learning_rate'] = 10**np.random.uniform(-5,-3)
    new_cfg['weight_decay'] = 10**np.random.uniform(-5,-3)
    new_cfg['batch_size'] = 2**np.random.randint(3,5)


    # featurizer-based cfg
    if new_cfg['featurizer'] == 'MLP':
        new_cfg['mlp_num_hidden'] = 2**np.random.randint(5,8)
        new_cfg['mlp_dropout'] = np.random.choice([0.0, 0.1, 0.5])
        new_cfg['mlp_width'] = 2**np.random.randint(6,10)
        new_cfg['mlp_depth'] = np.random.randint(3,5)
    
    elif new_cfg['featurizer'] == 'ResNet':
        new_cfg['resnet_dropout'] = np.random.choice([0.0, 0.1, 0.5])


    # algo-based cfg
    if new_cfg['algorithm'] == 'DANN':
        new_cfg['lambd'] = 10**np.random.uniform(-2, 2)
        new_cfg['d_steps_per_g_step'] = 2**np.random.uniform(0, 3)

    return new_cfg










