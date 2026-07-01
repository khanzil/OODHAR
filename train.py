from utils.cmd_parser import get_agrs_parser
from initialize import init_loader, init_algo
import os

def main():
    cfgs, args = get_agrs_parser()

    if args.mode == 'train':
        loaders, results_dir, ckpts_dir = init_loader(cfgs, args)

        for i_loader, (train_loader, in_val_loader, out_val_loader) in enumerate(loaders):
            print(f"Test dom no {i_loader}")
            algo = init_algo(cfgs, args)
            num_steps = cfgs['num_steps']
            dom_results_dir = os.path.join(results_dir, f'test_dom_{i_loader}')
            dom_ckpts_dir = os.path.join(ckpts_dir, f'test_dom_{i_loader}')

            if not os.path.isdir(dom_results_dir):
                os.makedirs(os.path.join(dom_results_dir,'ckpts'), exist_ok=True)
            if not os.path.isdir(dom_ckpts_dir):
                os.makedirs(os.path.join(dom_ckpts_dir,'ckpts'), exist_ok=True)

            loss_list = algo.train(num_steps=num_steps, 
                                   train_loader=train_loader, 
                                   in_val_loader=in_val_loader, 
                                   out_val_loader=out_val_loader, 
                                   results_dir=dom_results_dir,
                                   ckpts_dir=dom_ckpts_dir,
                                   val_freq=cfgs['val_freq'],
                                   ckpt_freq=cfgs['ckpt_freq'],
                                   test_dom=i_loader)

if __name__ == '__main__':
    main()




