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
        loss_list = algo.train(cur_epoch=cur_epoch, 
                               num_epochs=num_epochs, 
                               train_loader=train_loader, 
                               val_loader=val_loader, 
                               results_dir=results_dir,
                               ckpt_start=0)
 

    if args.mode == 'test': # for testing process, all predictions will be save for future use if needed 
        algo, test_loader, results_dir = init_test(cfg, args)
        infer_dict = {'acc' : 0, 'pred' : [], 'all_y' : []}

        pred_list = algo.validate_step(test_loader, trainval='val')

        infer_dict['pred'] = [int(pred) for pred, _ in pred_list]
        infer_dict['all_y'] = [int(y) for _, y in pred_list]
        infer_dict['acc'] = algo.loss_dict['val']['acc']
        print(infer_dict['pred'])

        output_file = open(os.path.join(results_dir, 'infer_dict'), 'w', encoding='utf-8')
        json.dump(infer_dict, output_file) 


if __name__ == '__main__':
    main()




