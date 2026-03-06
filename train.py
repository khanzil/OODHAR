from tqdm import tqdm
from utils.cmd_parser import get_agrs_parser
from initialize import init_train, init_test
import json
import os

def main():
    cfg, args = get_agrs_parser()

    if args.mode == 'train':
        algo, loaders, results_dir = init_train(cfg, args)

        if cfg['load_checkpoint'] == 'None':
            cur_epoch = 0
        else:
            cur_epoch = algo.load_ckpt(cfg['load_checkpoint'])
        
        for i_loader, (train_loader, val_loader, test_loader) in enumerate(loaders):
            num_epochs = cfg['num_epochs']
            dom_results_dir = os.path.join(results_dir, f'test_dom_{i_loader}')
            if not os.path.isdir(dom_results_dir):
                os.makedirs(os.path.join(dom_results_dir,'ckpts'), exist_ok=True)

            loss_list = algo.train(cur_epoch=cur_epoch, 
                                num_epochs=num_epochs, 
                                train_loader=train_loader, 
                                val_loader=val_loader, 
                                test_loader=test_loader,
                                results_dir=dom_results_dir,
                                ckpt_freq=cfg['ckpt_freq'])

if __name__ == '__main__':
    main()




