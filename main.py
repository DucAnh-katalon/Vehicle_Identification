import torch.optim as optim
from utils import *
from ops import *
from data import data_transforms,label_transform, Standford_Cars_Dataset
import deeplake
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
import time
import argparse

def parser():
    parser = argparse.ArgumentParser(description='NA')
    parser.add_argument('-config', '--config_dir', default='config.yaml', help='data version')
    args = parser.parse_args()
    return args

def train_model(model, dataloaders, criterion, optimizer, cfg):
    since = time.time()
    num_epochs = cfg['NUM_EPOCH']
    val_acc_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epoch_pbar = tqdm(range(num_epochs))

    for epoch in epoch_pbar:
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            iter_pbar = tqdm(dataloaders[phase])
            for inputs, labels in iter_pbar:
                inputs = inputs.cuda()
                labels = labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics 
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            epoch_pbar.set_postfix({
                        "loss": epoch_loss,
                        "acc": epoch_acc.item()
                        })            
        
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)    

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


if __name__=='__main__':
    args = parser() 
    cfg = get_config(config_path=args.config)

    ds_dict ={
        split : Standford_Cars_Dataset(data_dir = cfg['data_dir'], transform = data_transforms[split if split == 'train' else 'test'], split =split)
            for split in ['train','val','test']
    }
    dataloaders_dict = {
        split : DataLoader(ds_dict[split] ,batch_size= cfg['BATCH_SIZE'],shuffle= split =='train')
        for split in ['train','val','test']
    }

    net = initialize_model(cfg)
    net.cuda()
    params_to_update = net.parameters()
    optimizer_ft = optim.SGD(params_to_update, lr=cfg['lr'], momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    model_ft, hist = train_model(net, dataloaders_dict, criterion, optimizer_ft,cfg)