import numpy as np

def get_random_search_configs(cfgs):
    # same for every configs
    cfgs['learning_rate'] = 5 * 10**np.random.uniform(-5,-4)
    cfgs['weight_decay'] = 10**np.random.uniform(-4,-2)
    # cfgs['batch_size'] = 2**np.random.randint(3,5) # because trainer using fixed batch-steps instead of fixed epoch, random batch_size may cause unfair comparisons

    # featurizer-based cfgs
    if cfgs['featurizer'] == 'MLP':
        cfgs['mlp_num_hidden'] = 2**np.random.randint(5,8)
        cfgs['mlp_dropout'] = float(np.random.choice([0.0, 0.1, 0.5]))
        cfgs['mlp_width'] = 2**np.random.randint(6,10)
        cfgs['mlp_depth'] = np.random.randint(3,5)
    
    elif cfgs['featurizer'] == 'ResNet':
        cfgs['resnet_dropout'] = float(np.random.choice([0.0, 0.1, 0.5]))

    # algo-based cfgs
    if cfgs['algorithm'] == 'DANN':
        cfgs['DANN']['lambd'] = 10**np.random.uniform(-3,-2)
        cfgs['DANN']['lambd_iter'] = int(np.random.uniform(500,700))

    elif cfgs['algorithm'] == 'IRM':
        cfgs['IRM']['iter'] = int(np.random.uniform(500,700))
        cfgs['IRM']['lambd'] = 10**np.random.uniform(-1,1)

    elif cfgs['algorithm'] == 'VRex':
        cfgs['VRex']['iter'] = int(np.random.uniform(500,700))
        cfgs['VRex']['lambd'] = 10**np.random.uniform(-1,1)

    elif cfgs['algorithm'] == 'Fish':
        cfgs['Fish']['lr_meta'] = 5 * 10**np.random.uniform(-2,-1)