import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

detail = "Algo" # "All" for all test_dom avg, "Search" for all search avg, "Seed" for all seed avg, "Algo" for all algo avg
plot_graph = False

root_dir = './results/Glasgow/Room'
algo_list = ['ERM', 'DANN', 'IRM', 'VRex', 'GroupDRO', 'SAM', 'Fish', 'Fishr', 'CFSM', 'CORAL', 'MMD']

# root_dir = './results/Glasgow/Age'
# algo_list = ['ERM', 'DANN', 'IRM', 'VRex', 'GroupDRO', 'SAM', 'Fish', 'Fishr', 'CFSM', 'CORAL', 'MMD']



def sort_key(k):
    base_order = ['loss_class', 'tr_avg_acc', 'val_avg_acc', 'test_acc']
    if k in base_order:
        return base_order.index(k)
    return len(base_order)

# in this script, 'algo', 'seed',... usually used instead of 'algo_result', 'seed_result',... in 'for' loops
class search_results():
    def __init__(self, algo, root_dir, add_loss_dict=None):
        self.seed_list = [fold for fold in os.listdir(os.path.join(root_dir,algo)) if os.path.isdir(os.path.join(root_dir,algo,fold))]
        self.algo = algo
        self.root_dir = root_dir
        self.result = [] # structure: list_of_seeds[list_of_searches[list_of_test_dom[dict_of_results{}]]]
        self.read_last_results(add_loss_dict)

    def read_last_results(self, add_loss_dict=None):
        for i_seed, seed in enumerate(self.seed_list):
            search_list = [fold for fold in os.listdir(os.path.join(self.root_dir,self.algo,seed)) if os.path.isdir(os.path.join(self.root_dir,self.algo,seed,fold))]
            self.result.append([])
            for i_search, search in enumerate(search_list):
                self.result[i_seed].append([])
                for test_dom in os.listdir(os.path.join(self.root_dir,self.algo,seed,search)):
                    dom_result = {'loss_class': 0,
                                'tr_avg_acc': 0,
                                'val_avg_acc': 0,
                                'test_acc': 0}
                    if add_loss_dict is not None:
                        dom_result.update(add_loss_dict)
                    result_list = []

                    if not os.path.isdir(os.path.join(self.root_dir,self.algo,seed,search,test_dom)):
                        continue
                    with open(os.path.join(self.root_dir,self.algo,seed,search,test_dom,'loss_list.json'), 'r') as file:
                        for line in file:
                            my_dict = json.loads(line)
                            result_list.append(my_dict)
                    
                    for key in result_list[-1].keys():
                        if key in dom_result.keys():
                            dom_result[key] += result_list[-1][key]
                        elif 'te_dom' in key:
                            dom_result['test_acc'] += result_list[-1][key]

                    dom_result = {k: dom_result[k] for k in sorted(dom_result.keys(), key=sort_key)}

                    self.result[i_seed][i_search].append(dom_result)
                   
    def get_search_avg(self, return_best=True):
        self.avg_result = [] # structure: list_of_seeds[list_of_searches[dict_of_avg{}]]
        if return_best:
            self.seed_best = [[] for _ in range(len(self.result))]

        for i_seed, _ in enumerate(self.result):
            self.avg_result.append([])
            best_val = 0.0
            for i_search, _ in enumerate(self.result[i_seed]):
                search_avg_result = {'loss_class': 0,
                                    'tr_avg_acc': 0,
                                    'val_avg_acc': 0,
                                    'test_acc': 0}
                for i_test_dom, dom_result in enumerate(self.result[i_seed][i_search]):
                    for key in search_avg_result.keys():
                        search_avg_result[key] += dom_result[key]

                for key in search_avg_result.keys():
                    search_avg_result[key] /= len(self.result[i_seed][i_search])
                
                self.avg_result[i_seed].append(search_avg_result)
                
                if return_best:
                    if search_avg_result['val_avg_acc'] > best_val:
                        best_val = search_avg_result['val_avg_acc']
                        self.seed_best[i_seed] = (i_search, search_avg_result)

    def get_algo_avg(self):
        if not hasattr(self, 'seed_best'):
            self.get_search_avg(return_best=True)

        if len(self.seed_best) == 1:
            _, algo_avg = self.seed_best[0]
        else:
            algo_avg = {'loss_class': [],
                        'tr_avg_acc': [],
                        'val_avg_acc': [],
                        'test_acc': []}
            for i_seed, (_, seed) in enumerate(self.seed_best):
                for key in seed.keys():
                    algo_avg[key].append(seed[key])

        return algo_avg

if __name__ == '__main__':

    if detail in ['All']:
        keys = ['seed', 'search', 'test_dom', 'loss_class', 'tr_avg_acc', 'val_avg_acc', 'test_acc']
    elif detail in ['Search']:
        keys = ['seed', 'search', 'loss_class', 'tr_avg_acc', 'val_avg_acc', 'test_acc']
    elif detail in ['Seed']:
        keys = ['seed', 'loss_class', 'tr_avg_acc', 'val_avg_acc', 'test_acc']
    elif detail in ['Algo']:
        keys = ['loss_class', 'tr_avg_acc', 'val_avg_acc', 'test_acc']

    print("algo".ljust(30), end="")
    for key in keys:
        print(f"{key}".ljust(15), end="")
    print("")

    for i_algo, algo in enumerate(algo_list):
        algo_results = search_results(algo,root_dir)
        algo_results.get_search_avg()

        if detail == 'All':
            for i_seed, seed in enumerate(algo_results.result):
                for i_search, search in enumerate(seed):
                    for i_test_dom, test_dom in enumerate(search):
                        print(f"{algo}".ljust(30), end="")
                        print(f"seed{i_seed}".ljust(15), end="")
                        print(f"search{i_search}".ljust(15), end="")
                        print(f"test_dom_{i_test_dom}".ljust(15), end="")

                        for key in test_dom.keys():
                            if 'acc' in key:
                                test_dom[key] = int(test_dom[key] * 1e4)/1.0e2
                                print(f"{test_dom[key]:<.2f}".ljust(15), end="")
                            else:
                                print(f"{test_dom[key]:<.6f}".ljust(15), end="")
                        print("")

        if detail == 'Search':
            for i_seed, seed in enumerate(algo_results.avg_result):
                for i_search, search in enumerate(seed):
                    print(f"{algo}".ljust(30), end="")
                    print(f"seed{i_seed}".ljust(15), end="")
                    print(f"search{i_search}".ljust(15), end="")

                    for key in search.keys():
                        if 'acc' in key:
                            search[key] = int(search[key] * 1e4)/1.0e2
                            print(f"{search[key]:<.2f}".ljust(15), end="")
                        else:
                            print(f"{search[key]:<.6f}".ljust(15), end="")
                    print("")

        if detail == 'Seed':
            for i_seed, (_, seed) in enumerate(algo_results.seed_best):
                print(f"{algo}".ljust(30), end="")
                print(f"seed{i_seed}".ljust(15), end="")

                for key in seed.keys():
                    if 'acc' in key:
                        print(f"{seed[key]:<.2f}".ljust(15), end="")
                    else:
                        print(f"{seed[key]:<.6f}".ljust(15), end="")
                print("")

        if detail == 'Algo':
            algo_avg = algo_results.get_algo_avg()

            print(f"{algo}".ljust(30), end="")
            for key in algo_avg.keys():
                if 'acc' in key:
                    print(f"{100*np.mean(algo_avg[key]):<.2f}\u00B1{100*np.std(algo_avg[key]):<.2f}".ljust(15), end="")
                else:
                    print(f"{np.mean(algo_avg[key]):<.4f}\u00B1{np.std(algo_avg[key]):<.4f}".ljust(15), end="")


            print("")


    #     if plot_graph:
    #         for i_seed, (i_search, _) in enumerate(algo_results.seed_best):
    #             search = algo_results.result[i_seed][i_search]
    #             plot_array = []
    #             for i_test_dom, test_dom in enumerate(search):

    #                 plot_array.append([test_dom[key] for key in ['tr_avg_acc', 'val_avg_acc', 'test_acc']])

    #             plot_array = np.stack(plot_array) # n_test_dom x 3
    #             x = np.arange(len(plot_array))
    #             width = 0.2
    #             for arr in plot_array:
    #                 ax[i_algo, i_seed].bar(x - width, plot_array[:,0], width, label='tr_avg_acc', color='#90be6d', edgecolor='black')
    #                 ax[i_algo, i_seed].bar(x        , plot_array[:,1], width, label='val_avg_acc', color="#ffad32", edgecolor='black')
    #                 ax[i_algo, i_seed].bar(x + width, plot_array[:,2], width, label='test_avg_acc', color="#487fff", edgecolor='black')


    #         ax[i_algo, i_seed].set_ylabel('Acc (%)', fontsize=14)
    #         ax[i_algo, i_seed].set_xticks(x)
    #         ax[i_algo, i_seed].set_xticklabels(x)

    #         ax[i_algo, i_seed].set_ylim(0.7,1.0)

            


    # if plot_graph:
    #     plt.tight_layout()
    #     plt.show()





