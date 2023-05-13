import torch.optim as optim
from Vehicle_Identification.utils import *
from Vehicle_Identification.ops import *
from Vehicle_Identification.data import data_transforms,label_transform
import deeplake
from tqdm import tqdm
import time



def train_model(model, dataloaders, criterion, optimizer, cfg):
    since = time.time()
    num_epochs = cfg['NUM_EPOCH']
    val_acc_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epoch_pbar = tqdm(range(num_epochs))

    for epoch in epoch_pbar:
        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
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
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'test':
                val_acc_history.append(epoch_acc)    

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


if __name__=='__main__':
    
    cfg = get_config()
    ds_dict = {x :deeplake.load(f"hub://activeloop/stanford-cars-{x}") for x in ['train','test']}
    dataloaders_dict = {
                    x: ds_dict[x].pytorch(num_workers=0, batch_size= cfg['BATCH_SIZE'],
                                        transform={'images': data_transforms[x], 'car_models':label_transform}, 
                                        collate_fn = custom_collate_fn, shuffle=True, 
                                        decode_method = {'images':'pil','car_models':'data'}) 
                    for x in ['train','test']
                    }
    
    net = initialize_model(cfg)
    net.cuda()
    params_to_update = net.parameters()
    optimizer_ft = optim.SGD(params_to_update, lr=cfg['lr'], momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    model_ft, hist = train_model(net, dataloaders_dict, criterion, optimizer_ft,cfg)