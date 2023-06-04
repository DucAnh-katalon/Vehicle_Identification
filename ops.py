
import time
import copy
import numpy as np
import torch.nn as nn
import timm
import torch 
import numpy
from tqdm import tqdm


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def custom_collate_fn(original_batch):
    labels = torch.tensor([ _[1] for _ in original_batch], dtype = int)
    imgs = torch.stack( [ _[0] for _ in original_batch])
    return imgs,labels


def initialize_model(cfg):
    
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None

    if cfg['model_name'] == "resnet50d":
        """
            Resnet50d
        """
        # model_ft = timm.create_model("seresnextaa101d_32x8d",
        #                              pretrained= True,
        #                              num_classes = cfg['NUM_CLASSES']
        #                             )
        model_ft = timm.create_model("resnet50d",
                                     pretrained= True,
                                     num_classes = cfg['NUM_CLASSES']
                                    )
    
    else:
        print("Invalid model name, exiting...")
        exit()
    
    return model_ft