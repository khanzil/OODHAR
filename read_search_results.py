import numpy as np
import os
import json

root_dir = './results/Glasgow/Old'
search_list = [fold for fold in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir,fold)) and 'seed' in fold]
search_results = []
feas = []
algos = []
for fold in search_list:
    idx = fold.split('_')
    algos.append(idx[2])
    feas.append(idx[3])
    dom_loss = {'seed': idx[0],
                'search': idx[1],
                'algo': idx[2],
                'featurizer': idx[3],
                'loss_class': 0.0,
                'train_acc': 0.0,
                'val_acc': 0.0,
                'test_acc': 0.0}
    for test_dom in os.listdir(os.path.join(root_dir,fold)):
        print(dom_loss['algo'], dom_loss['featurizer'], test_dom)
        
        loss_list = []
        if not os.path.isdir(os.path.join(root_dir,fold,test_dom)):
            continue
        with open(os.path.join(root_dir,fold,test_dom,'loss_list'), 'r') as file:
            for line in file:
                loss_list.append(json.loads(line))

        for key in ['loss_class', 'train_acc', 'val_acc', 'test_acc']:
            dom_loss[key] += loss_list[-1][key]
            print({loss_list[-1][key]})


    for key in ['loss_class', 'train_acc', 'val_acc', 'test_acc']:
        dom_loss[key] = dom_loss[key]/7.0


    search_results.append(dom_loss)

keys = ['seed', 'search', 'algo', 'featurizer', 'loss_class', 'train_acc', 'val_acc', 'test_acc']

for key in keys:
    print(f"{key}".ljust(15), end="")
print("")

for fea in np.unique(feas):
    for algo in np.unique(algos):
        for search in search_results:
            if search['featurizer'] != fea or search['algo'] != algo:
                continue

            for key in search.keys():
                if key in ['loss_class', 'train_acc', 'val_acc', 'test_acc']:
                    print(f"{search[key]:.10f}".ljust(15), end="")
                else:
                    print(f"{search[key]}".ljust(15), end="")
            print("")














