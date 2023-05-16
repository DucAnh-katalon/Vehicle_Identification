from torch.utils.data import Dataset, DataLoader
import deeplake
from deeplake.core.sample import Sample
from torchvision import datasets, models, transforms
import torch
from pathlib import Path
from tqdm import tqdm
import io,os
from PIL import Image
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
    'val':lambda im: 
        transforms.Compose([
            transforms.Resize((224,224)),
            # transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))  if im.mode!='RGB'  else NoneTransform(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])(im),
}

class Standford_Cars_Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, ds, transform=None,data_dir = './data', extract_data = False):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.ds = ds
        self.data_dir = Path(data_dir)

        if not os.path.exists(data_dir) or extract_data:
            print("extracting....")
            os.makedirs(self.data_dir, exist_ok=True)
            for idx,entry in tqdm(enumerate(self.ds),total = len(self.ds)):
                label = entry['car_models'].data()['value'][0]
                descript_label = entry['car_models'].data()['text'][0]
                img_data = Sample(array = entry['images'].data()['value']).compressed_bytes(compression='jpeg')

                label_dir = self.data_dir / str(label)
                if not os.path.exists(label_dir):
                    os.mkdir(label_dir)
                    open(label_dir / 'description.txt','w').write(descript_label)

                fn = label_dir / f'{idx}.jpeg'
                Image.open(io.BytesIO(img_data)).save(fn)
        
        self.images = []
        self.labels = []
        for fn in self.data_dir.glob("*/*.jpeg"):
            self.images.append(fn)
            self.labels.append(fn.parent.stem)

        # self.labels = torch.tensor(self.ds.tensors['car_models'].data()['value'].reshape(-1,).astype(int))
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = Image.open(self.images[idx])
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)

        return sample,label