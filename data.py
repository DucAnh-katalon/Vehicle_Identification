from torch.utils.data import Dataset, DataLoader
import deeplake
from deeplake.core.sample import Sample
from torchvision import datasets, models, transforms
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import io,os
from PIL import Image
from timm.data.transforms_factory import create_transform 
from timm.data.auto_augment import rand_augment_transform
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
    'train': create_transform(
        input_size=224,
        is_training=True,
        auto_augment='rand-m5-mstd0.2',
    ),
    'test':create_transform(
        input_size=224,
    )

}




class Standford_Cars_Dataset(Dataset):
    """Standford Car dataset."""

    def __init__(self, data_dir = './data',split ='train', transform = None, extract_data = False):
        self.data_dir = Path(data_dir) / split
        if not os.path.exists(data_dir) or extract_data:
            extract_data_from_deeplake(self.data_dir.parent)
        
        self.images = []
        self.labels = []
        for fn in self.data_dir.glob("*/*.jpeg"):
            self.images.append(fn)
            self.labels.append(fn.parent.stem)

        # self.labels = torch.tensor(self.ds.tensors['car_models'].data()['value'].reshape(-1,).astype(int))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        sample = Image.open(self.images[idx]).convert('RGB')
        label = int(self.labels[idx])
        if self.transform:            
            sample = self.transform(sample)
        return sample,label

def extract_data_from_deeplake(data_dir:Path, val_set_ratio:float= 0.2):
    print("extracting....")
    os.makedirs(data_dir, exist_ok=True)
    ds_dict ={split: deeplake.load(f"hub://activeloop/stanford-cars-{split}") for split in ['train','test']}
    
    if val_set_ratio > 0:
        val_idx = random_val_idx(ds_dict['train'],val_set_ratio )
        os.makedirs(data_dir / 'val', exist_ok=True)
        
    for split, ds in ds_dict.items():
        os.makedirs(data_dir / split, exist_ok=True)
        print(f"Extracting {split} set")
        for idx,entry in tqdm(enumerate(ds),total = len(ds)):
            label = entry['car_models'].data()['value'][0]
            descript_label = entry['car_models'].data()['text'][0]
            img_data = Sample(array = entry['images'].data()['value']).compressed_bytes(compression='jpeg')
            if (idx not in val_idx) or (split =='test'):
                label_dir = data_dir/ split /str(label) 
            else:
                label_dir = data_dir / 'val' /str(label)        
            if not os.path.exists(label_dir):
                os.mkdir(label_dir)
                open(label_dir / 'description.txt','w').write(descript_label)

            fn = label_dir / f'{idx}.jpeg'
            Image.open(io.BytesIO(img_data)).save(fn)

def random_val_idx(ds, val_set_ratio) ->list:
    idx_df  = pd.DataFrame(ds['car_models'].data()['value'],columns =['label']).reset_index()
    val_idx = []
    for  _,gr in idx_df.groupby('label'):
        val_size = int(gr.shape[0] * val_set_ratio)
        val_idx += gr.sample(val_size)['index'].values.tolist()
    return val_idx