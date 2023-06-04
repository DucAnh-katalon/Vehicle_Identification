
import torch.optim as optim
from utils import *
from ops import *
from data import data_transforms,label_transform, Standford_Cars_Dataset
import deeplake
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
import time
import argparse

import wandb
wandb_api = "86b9a5c5a2b9ad64302c105c8653d9a58e7552fc"
wandb.login(key=wandb_api)


def parser():
    parser = argparse.ArgumentParser(description='NA')
    parser.add_argument('-config', '--config_path', default='config.yaml', help='data version')
    args = parser.parse_args()
    return args

def train_model(model, dataloaders, criterion, optimizer, cfg):
    run = wandb.init(project='standfor_cars', 
                 config=cfg,
                 group='resnet50d', 
                 job_type='train')
    
    since = time.time()
    artifact = wandb.Artifact(name=cfg['model_name'], type='model')
    cooldown_epochs = cfg['warm_up']
    num_epochs = cfg['NUM_EPOCH'] + cooldown_epochs
    val_acc_history = []
    scheduler = timm.scheduler.CosineLRScheduler(optimizer, t_initial=cfg['NUM_EPOCH'])
    
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
            if (phase == 'val') and ((epoch % 5) !=0):
                continue
                
                
            running_loss = 0.0
            running_corrects = 0
            num_steps_per_epoch = len(dataloaders[phase])
            num_updates = epoch * num_steps_per_epoch
            
            # Iterate over data.
            for idx, (inputs, labels) in enumerate(dataloaders[phase]):
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
                        scheduler.step_update(num_updates=num_updates)
        
                # statistics 
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)                
                   
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = (running_corrects.double() / len(dataloaders[phase].dataset)).item()

            log_epoch = {
                f"{phase}_loss": epoch_loss,
                f"{phase}_acc" : epoch_acc 
            }
            epoch_pbar.set_postfix(log_epoch)
            wandb.log(log_epoch)
            
            
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
#                 best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), f"{cfg['output']}_best.pt")
            if phase == 'val':
                val_acc_history.append(epoch_acc)  
        scheduler.step(epoch + 1)



    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    torch.save(model.state_dict(), f"{cfg['output']}_last.pt")
    artifact.add_file(f"{cfg['output']}_last.pt")
    artifact.add_file(f"{cfg['output']}_best.pt")
    run.log_artifact(artifact)
    run.finish()        
    model.load_state_dict(torch.load(f"{cfg['output']}_best.pt"))  
    return model, val_acc_history



if __name__=='__main__':
    args = parser() 
    cfg = get_config(config_path=args.config_path)

    ds_dict ={
    split : Standford_Cars_Dataset(data_dir = cfg['data_dir'], transform = data_transforms[split if split == 'train' else 'test'], split =split)
        for split in ['train','val','test']
    }
    dataloaders_dict = {
        split : DataLoader(ds_dict[split] ,batch_size= cfg['BATCH_SIZE'],shuffle= split =='train')
        for split in ['train','val','test']
    }


    net = initialize_model(cfg)

    #l load checkpoint 
    # artifact = run.use_artifact('resnet50d:lastest')
    # datadir = artifact.download()
    # net.load_state_dict(torch.load(f'{datadir}/_best.pt'))

    net.cuda()
    params_to_update = net.parameters()
    optimizer = timm.optim.AdamP(params_to_update, lr = cfg['lr'], weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()


    # train 
    wandb.config.type = 'baseline'
    model_ft, hist = train_model(net, dataloaders_dict, criterion, optimizer,cfg)
    


    # evaluate test set
    print('Evaluating test set')
    model_ft.eval()
    model_ft.cuda()
    running_loss = 0.0
    running_corrects = 0
    num_steps_per_epoch = len(dataloaders_dict['test'])


    # Iterate over data.
    for idx, (inputs, labels) in enumerate(dataloaders_dict['test']):
        inputs = inputs.cuda()
        labels = labels.cuda()
        # zero the parameter gradients
        with torch.set_grad_enabled(False):
            outputs = model_ft(inputs)
            _, preds = torch.max(outputs, 1)
        # statistics 
        running_corrects += torch.sum(preds == labels.data)   

    epoch_acc = (running_corrects.double() / len(dataloaders_dict['test'].dataset)).item()
    print(f"Test acc: {epoch_acc}")

    