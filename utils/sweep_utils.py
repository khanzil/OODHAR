import numpy as np
import copy

def get_random_search_configs(cfgs, seed, search, algo, featurizer):
    new_cfgs = copy.deepcopy(cfgs)
    new_cfgs['train_id'] = f'seed{seed}_search{search}_{algo}_{featurizer}'
    new_cfgs['algorithm'] = algo
    new_cfgs['featurizer'] = featurizer
    

    # same for every configs
    new_cfgs['learning_rate'] = 5 * 10**np.random.uniform(-5,-4)
    new_cfgs['weight_decay'] = 10**np.random.uniform(-5,-3)
    new_cfgs['batch_size'] = 2**np.random.randint(3,5)


    # featurizer-based cfgs
    if new_cfgs['featurizer'] == 'MLP':
        new_cfgs['mlp_num_hidden'] = 2**np.random.randint(5,8)
        new_cfgs['mlp_dropout'] = float(np.random.choice([0.0, 0.1, 0.5]))
        new_cfgs['mlp_width'] = 2**np.random.randint(6,10)
        new_cfgs['mlp_depth'] = np.random.randint(3,5)
    
    elif new_cfgs['featurizer'] == 'ResNet':
        new_cfgs['resnet_dropout'] = float(np.random.choice([0.0, 0.1, 0.5]))

    # algo-based cfgs
    if new_cfgs['algorithm'] == 'DANN':
        new_cfgs['lambd'] = 10**np.random.uniform(-3, -1)
        new_cfgs['d_steps_per_g_step'] = int(2**np.random.uniform(0, 3))

    return new_cfgs










