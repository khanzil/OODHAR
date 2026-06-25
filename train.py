from utils.cmd_parser import get_agrs_parser
from initialize import init_train, init_algo
import os

def main():
    cfgs, args = get_agrs_parser()

    if args.mode == 'train':
        algo, loaders, results_dir, ckpts_dir = init_train(cfgs, args)

        if cfgs['load_checkpoint'] == 'None':
            cur_step = 0
        else:
            cur_step = algo.load_ckpt(cfgs['load_checkpoint'])
        
        for i_loader, (train_loader, in_val_loader, out_val_loader, test_loader) in enumerate(loaders):
            print(f"Test dom no {i_loader}")
            algo = init_algo(cfgs, args)
            num_steps = cfgs['num_steps']
            dom_results_dir = os.path.join(results_dir, f'test_dom_{i_loader}')
            dom_ckpts_dir = os.path.join(ckpts_dir, f'test_dom_{i_loader}')

            if not os.path.isdir(dom_results_dir):
                os.makedirs(os.path.join(dom_results_dir,'ckpts'), exist_ok=True)
            if not os.path.isdir(dom_ckpts_dir):
                os.makedirs(os.path.join(dom_ckpts_dir,'ckpts'), exist_ok=True)

            loss_list = algo.train(cur_step=cur_step, 
                                num_steps=num_steps, 
                                train_loader=train_loader, 
                                in_val_loader=in_val_loader, 
                                out_val_loader=out_val_loader, 
                                test_loader=test_loader,
                                results_dir=dom_results_dir,
                                ckpts_dir=dom_ckpts_dir,
                                val_freq=cfgs['val_freq'],
                                ckpt_freq=cfgs['ckpt_freq'])

if __name__ == '__main__':
    main()




