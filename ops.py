
import time
import copy
import numpy as np
import torch.nn as nn

import torch 
import numpy
from tqdm import tqdm


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def custom_collate_fn(original_batch):
    labels = torch.tensor([ _['car_models'] for _ in original_batch], dtype = int)
    imgs = torch.stack( [ _['images'] for _ in original_batch])
    return imgs,labels

def initialize_model(cfg):
    
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None

    if cfg['model_name'] == "resnet18":
        """
            Resnet18
        """
        from torchvision.models import resnet18, ResNet18_Weights
        model_ft = resnet18(weights=ResNet18_Weights.DEFAULT)
        set_parameter_requires_grad(model_ft, cfg['feature_extract'])
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, cfg['NUM_CLASSES'])
        
    else:
        print("Invalid model name, exiting...")
        exit()
    
    return model_ft





