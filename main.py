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
            algo.init_loss_dict(trainval='all') # reset loss_dict, do nothing if loss_dict dont exist
            '''
                Perform training
            '''
            algo.train_step(train_loader)

            '''
                Calculate metrics on validation set and train. Train loss is already calculated during training
            '''
            algo.validate_step(train_loader, trainval='train')
            algo.validate_step(val_loader, trainval='val')

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
        

        infer_dict = {'acc' : 0, 'pred' : [], 'all_y' : []}

        pred_list = algo.validate_step(test_loader, trainval='val')

        infer_dict['pred'] = [pred for pred, _ in pred_list]
        infer_dict['all_y'] = [y for _, y in pred_list]
        infer_dict['acc'] = algo.loss_dict['val']['acc']

        output_file = open(os.path.join(results_dir, 'infer_dict'), 'w', encoding='utf-8')
        print(infer_dict['acc'])
        json.dump(infer_dict, output_file) 


if __name__ == '__main__':
    main()




