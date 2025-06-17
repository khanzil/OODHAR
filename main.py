from tqdm import tqdm
from utils.cmd_parser import get_agrs_parser
from initialize import init_train


def main():
    cfg, args = get_agrs_parser()
    algo, train_loader, val_loader = init_train(cfg, args)
    if cfg['load_checkpoint'] is None:
        cur_epoch = 0
    else:
        cur_epoch = algo.load_ckpt(cfg['load_checkpoint'])
    
    num_epochs = cfg['train']['num_epochs']

    for epoch in range(num_epochs-cur_epoch):
        iterator = tqdm(train_loader, total=len(train_loader), unit='it')
        for batch_idx, minibatch in enumerate(iterator):
            train_losses = algo.update(minibatch)

        '''
            Calculate loss on validation set
        '''
        iterator = tqdm(val_loader, total=len(val_loader), unit='it')

        for batch_idx, minibatch in enumerate(iterator):
            val_losses = algo.validate(minibatch)




if __name__ == '__main__':
    main()




