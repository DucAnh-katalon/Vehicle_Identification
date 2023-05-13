from torch.utils.data import Dataset, DataLoader
import deeplake
from torchvision import datasets, models, transforms
import torch

# Data augmentation and normalization for training
# Just normalization for validation
class NoneTransform(object):
    ''' Does nothing to the image. To be used instead of None '''
    
    def __call__(self, image):       
        return image


def label_transform(cars_model):
    # new_map = dict( (v,k) for k,v in enumerate(set([" ".join(_[0].split()[:-1]) for _ in ds_dict['train']['car_models'].data()['text']])))
    # car_label_transform = lambda x: new_map[" ".join(x['text'][0].split()[:-1])]
    return cars_model['value'][0]
    
data_transforms = {
    'train': lambda im:  
        transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))  if im.mode!='RGB'  else NoneTransform()  ,
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])(im),
    'test':lambda im: 
        transforms.Compose([
            transforms.Resize((224,224)),
            # transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))  if im.mode!='RGB'  else NoneTransform(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])(im),
}

