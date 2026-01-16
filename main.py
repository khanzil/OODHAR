from tqdm import tqdm
from utils.cmd_parser import get_agrs_parser
from initialize import init_train, init_test
import json
import os

def main():
    cfg, args = get_agrs_parser()
    if args.mode == 'train':
        algo, train_loader, val_loader, results_dir = init_train(cfg, args)

        if cfg['load_checkpoint'] == 'None':
            cur_epoch = 0
        else:
            cur_epoch = algo.load_ckpt(cfg['load_checkpoint'])
        
        num_epochs = cfg['train']['num_epochs']
        loss_list = []

        for epoch in range(num_epochs-cur_epoch):
            algo.init_loss_dict() # reset loss_dict, do nothing if loss_dict dont exist
            
            
            '''
                Perform training
            '''
            iterator = tqdm(train_loader, total=len(train_loader), unit='batch', position=0, leave=True)
            for batch_idx, minibatch in enumerate(iterator):
                algo.update(minibatch)
                algo.validate(minibatch,istrain=True)


            '''
                Calculate loss on validation set
            '''
            iterator = tqdm(val_loader, total=len(val_loader), unit='batch', position=0, leave=True)
            for batch_idx, minibatch in enumerate(iterator):
                algo.validate(minibatch)

            for train_val in algo.loss_dict.keys():
                for key in algo.loss_dict[train_val].keys():
                    if key == 'loader_len':
                        continue
                    algo.loss_dict[train_val][key] = algo.loss_dict[train_val][key]/algo.loss_dict[train_val]['loader_len']
            
            loss_list.append(algo.loss_dict) # add validation results to loss_list


            '''
                Print and save validation results after every epoch
            '''
            print(f'Epoch {epoch+1}/{num_epochs}: ')
            for train_val in loss_list[-1].keys():
                print(f'{train_val}: ', end="")
                for key in loss_list[-1][train_val].keys():
                    print(f'{key}: {loss_list[-1][train_val][key]:.5f},  ', end="")
                print("")
            output_file = open(os.path.join(results_dir, 'loss_list'), 'a', encoding='utf-8')
            for dic in loss_list:
                json.dump(dic, output_file)
                output_file.write("\n")


            '''
                Save the last 15% checkpoints and the best one
            '''
            if epoch >= 0.85*num_epochs: # save last 15% check_points
                algo.save_ckpt(epoch, results_dir)
            

    if args.mode == 'test': # for testing process, all predictions will be save for future use if needed 
        algo, test_loader, results_dir = init_test(cfg, args)
        
        iterator = tqdm(test_loader, total=len(test_loader), unit='batch', position=0, leave=True)
        infer_dict = {'acc' : 0, 'pred' : [], 'all_y' : []}

        for batch_idx, minibatch in enumerate(iterator):
            pred, y = algo.validate(minibatch)
            infer_dict['pred'].extend(pred.tolist())
            infer_dict['all_y'].extend(y.tolist())
        infer_dict['n_correct'] =float(algo.loss_dict['val']['n_correct'])/float(algo.loss_dict['val']['loader_len'])
        output_file = open(os.path.join(results_dir, 'infer_dict'), 'w', encoding='utf-8')
        print(infer_dict['n_correct'])
        json.dump(infer_dict, output_file) 


if __name__ == '__main__':
    main()




