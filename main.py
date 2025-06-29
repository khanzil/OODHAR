from tqdm import tqdm
from utils.cmd_parser import get_agrs_parser
from initialize import init_train
import pandas

def main():
    cfg, args = get_agrs_parser()
    algo, train_loader, val_loader, results_dir = init_train(cfg, args)
    # loss_csv = [[key for key in algo.loss_dict]]
    # for key in algo.loss_dict_val:
    #     loss_csv[0].append(key)
    # loss_csv.append('acc')

    if cfg['load_checkpoint'] == 'None':
        cur_epoch = 0
    else:
        cur_epoch = algo.load_ckpt(cfg['load_checkpoint'])
    
    num_epochs = cfg['train']['num_epochs']

    for epoch in range(num_epochs-cur_epoch):
        train_loss = []
        val_loss = []
        
        iterator = tqdm(train_loader, total=len(train_loader), unit='batch', position=0, leave=True)
        for batch_idx, minibatch in enumerate(iterator):       
            algo.update(minibatch)
            train_loss.append(algo.loss_dict)

        algo.save_ckpt(epoch, results_dir)
        '''
            Calculate loss on validation set
        '''
        iterator = tqdm(val_loader, total=len(val_loader), unit='batch', position=0, leave=True)
        num_corrects = 0
        num_samples = 0
        for batch_idx, minibatch in enumerate(iterator):
            num_samples += minibatch.batch_feature.shape[0]
            num_corrects += algo.validate(minibatch)
            val_loss.append(algo.loss_dict_val)

        print(f'Epoch {epoch+1}/{num_epochs}: ')

        print('Train: ', end="")
        for key in train_loss[-1].keys():
            print(f'{key}: {train_loss[-1][key]:.5f},  ', end="")
        print("")

        print('Val:  ', end="")
        for key in val_loss[-1].keys():
            print(f'{key}: {val_loss[-1][key]:.5f},  ', end="")
        print(f'acc: {num_corrects/num_samples:.5f}')
        

if __name__ == '__main__':
    main()




